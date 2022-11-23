import * as cdk from 'aws-cdk-lib';
import { Construct } from 'constructs';
// import * as sqs from 'aws-cdk-lib/aws-sqs';
import * as ec2 from 'aws-cdk-lib/aws-ec2';
import * as ecsPatterns from 'aws-cdk-lib/aws-ecs-patterns';
import * as ecs from 'aws-cdk-lib/aws-ecs';
import * as elbv2 from 'aws-cdk-lib/aws-elasticloadbalancingv2';

const EPHEMERAL_PORT_RANGE = ec2.Port.tcpRange(32768, 65535);


export class TestStack extends cdk.Stack {
  constructor(scope: Construct, id: string, props?: cdk.StackProps) {
    super(scope, id, props);

    // dl1 instance type needs to be available
    // aws ec2 describe-instance-type-offerings --location-type "availability-zone" --filters Name=instance-type,Values=dl1.24xlarge --region us-east-1 --query "InstanceTypeOfferings[*].[Location]" --output text | sort
    const vpc = new ec2.Vpc(this, 'MyVpc', { availabilityZones: ['us-east-1b','us-east-1c'] });
    const cluster = new ecs.Cluster(this, 'Ec2Cluster', { vpc });


    // https://aws.amazon.com/marketplace/pp/prodview-h24gzbgqu75zq?sr=0-6&ref_=beagle&applicationId=AWSMPContessa
    const habanaImage = new ec2.GenericLinuxImage({
      'us-east-1': 'ami-0fae7eebcc0d5c84f',
    });

    cluster.addCapacity('DefaultAutoScalingGroup', {
      instanceType: new ec2.InstanceType("dl1.24xlarge"),
      machineImage: habanaImage
    });


    const loadBalancedEcsService = new ecsPatterns.ApplicationLoadBalancedEc2Service(this, 'Service', {
      cluster,
      memoryLimitMiB: 760000,
      cpu: 98000,
      desiredCount: 1,
      taskImageOptions: {
        image: ecs.ContainerImage.fromRegistry('huggingface/optimum-habana:latest'),
        environment: {
          "HABANA_VISIBLE_DEVICES": "all"
        },
        // entryPoint: ['/bin/bash', '-c'],
        // command: ['hl-smi'],
      },
    });
  }
}



