import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
// import * as sqs from 'aws-cdk-lib/aws-sqs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecsPatterns from 'aws-cdk-lib/aws-ecs-patterns';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';
import * as path from 'path';

const EPHEMERAL_PORT_RANGE = ec2.Port.tcpRange(0, 65535);

export class EcsClusterStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // dl1 instance type needs to be available
    // aws ec2 describe-instance-type-offerings --location-type "availability-zone" --filters Name=instance-type,Values=dl1.24xlarge --region us-east-1 --query "InstanceTypeOfferings[*].[Location]" --output text | sort
    const vpc = new ec2.Vpc(this, 'MyVpc', { availabilityZones: ['us-east-1b', 'us-east-1c'] });
    const cluster = new ecs.Cluster(this, 'Ec2Cluster', { vpc });

    // https://aws.amazon.com/marketplace/pp/prodview-h24gzbgqu75zq?sr=0-6&ref_=beagle&applicationId=AWSMPContessa
    const habanaImage = new ec2.GenericLinuxImage({
      'us-east-1': 'ami-0fae7eebcc0d5c84f',
    });

    cluster.addCapacity('DefaultAutoScalingGroup', {
      instanceType: new ec2.InstanceType("dl1.24xlarge"),
      machineImage: habanaImage
    });

    const hub_token = process.env.HF_HUB_TOKEN || undefined

    const loadBalancedEcsService = new ecsPatterns.NetworkLoadBalancedEc2Service(this, 'Service', {
      cluster,
      memoryLimitMiB: 760000,
      cpu: 98000,
      desiredCount: 1,
      taskImageOptions: {
        image: new ecs.AssetImage(path.join(__dirname, '..','..','app'), {
          file: 'container/Dockerfile',
        }),
        environment: {
          "HABANA_VISIBLE_DEVICES": "all",
          // "HF_HUB_TOKEN": hub_token // not longer needed
        },
        containerPort: 8000,
        // entryPoint: ["/bin/bash", "-c"],
        // command: ["hl-smi"]
      },
    });
        // Need target security group to allow all inbound traffic for
    // ephemeral port range (when host port is 0).
    loadBalancedEcsService.service.connections.allowFromAnyIpv4(EPHEMERAL_PORT_RANGE);
    loadBalancedEcsService.targetGroup.configureHealthCheck({
      interval: cdk.Duration.seconds(30),
      unhealthyThresholdCount: 10,
    });


    // Outputs
    new cdk.CfnOutput(this, "networkLoadBalancerURL", {
      value: "http://"+loadBalancedEcsService.loadBalancer.loadBalancerDnsName,
      description: "Network LoadBalancer URL"
    });
  }
}



