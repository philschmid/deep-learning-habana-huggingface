# Getting Started with Deep Learning on Habana Gaudi with Hugging Face

This repository contains instructions for getting started with Deep Learning on Habana Gaudi with Hugging Face libraries like [transformers](https://huggingface.co/docs/transformers/index), [optimum](https://huggingface.co/docs/optimum/index), [datasets](https://huggingface.co/docs/datasets/index). This guide will show you how to set up the develoment environment on the AWS cloud and get started with Hugging Face Libraries. It doesn't contain a detailed guide on how to fine-tune models. This is covered in this [post](https://todo.com).

This guide will cover:
1. Requirements
2. Create an AWS EC2 instance
3. Connect to the instance via ssh
4. Connect to the instance via vscode server


## Requirements

* AWS Account with quota for [DL1 instance type](https://aws.amazon.com/ec2/instance-types/dl1/)
* [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) installed
* AWS IAM user configured with permission to create and manage ec2 instances


## Create an AWS EC2 instance

Before we can launch or AWS DL1 instance we have to create a `key-pair` and `security-group`, which we will use to access the instance.

Configure AWS PROFILE and AWS REGION which will be used for the instance

```bash
export AWS_PROFILE=<your-aws-profile>
export AWS_DEFAULT_REGION=<your-aws-region>
```

First we create a key pair, which will be used later to ssh into the instance. 
```bash
KEY_NAME=habana
aws ec2 create-key-pair --key-name ${KEY_NAME} --query 'KeyMaterial' --output text > ${KEY_NAME}.pem
chmod 400 ${KEY_NAME}.pem
```
  
Then we create a security group, which allows SSH access to the instance. We are going to use the default VPC for it, but can adjust the `vpc-id` arg if you want to modify it.
```bash
SG_NAME=habana
DEFAULT_VPC_ID=$(aws ec2 describe-vpcs --query 'Vpcs[?isDefault==true].VpcId' --output text)
echo "Default VPC ID: ${DEFAULT_VPC_ID}"
SECURITY_GROUP_ID=$(aws ec2 create-security-group --group-name ${SG_NAME}-sg --description "SG for Habana Deep Learning" --vpc-id ${DEFAULT_VPC_ID} --output text)
echo "Security Group ID: ${SECURITY_GROUP_ID}"
echo $(aws ec2 authorize-security-group-ingress --group-id ${SECURITY_GROUP_ID} --protocol tcp --port 22 --cidr 0.0.0.0/0 --output text)
```

After we have successfully made sure that we can use the Gaudi machine in a secure environment it is time to create the instance.
  
To use the official marketplace image you have to subscribe on the UI first and then you can access it with the following command `AMI_ID=$(aws ec2 describe-images --filters "Name=name,Values=* Habana Deep Learning Base AMI (Ubuntu 20.*"  --query 'Images[0].ImageId' --output text)`. In the guide we are using the community version, which is the exact same version as the official one.

```bash
AMI_ID=$(aws ec2 describe-images --filters "Name=name,Values=*habanalabs-base-ubuntu20.04*"  --query 'Images[0].ImageId' --output text)
echo "AMI ID: ${AMI_ID}"
INSTANCE_TYPE=dl1.24xlarge
INSTANCE_NAME=habana

aws ec2 run-instances \
  --image-id ${AMI_ID} \
  --key-name ${KEY_NAME} \
  --count 1 \
  --instance-type ${INSTANCE_TYPE} \
  --security-group-ids ${SECURITY_GROUP_ID} \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME}-demo}]"
```

Now, the instance is running and we can connect to it via ssh.

```bash
PUBLIC_DOMAIN=$(aws ec2 describe-instances --profile sandbox \
    --filters Name=tag-value,Values=${INSTANCE_NAME}-demo  \
    --query 'Reservations[*].Instances[*].PublicDnsName' \
    --output text)
ssh -i ${KEY_NAME}.pem ubuntu@${PUBLIC_DOMAIN//[$'\t\r\n ']}
```

Lets see if we can access the Gaudi devices. Habana provides a similar CLI tool like `nvidia-smi` with `hl-smi` command.
You can find more documentation [here](https://docs.habana.ai/en/latest/Management_and_Monitoring/System_Management_Tools_Guide/System_Management_Tools.html).
```bash
hl-smi
```
You should a similar output that below.

![hl-smi](assets/hl-smi.png)

We can also test if we can allocate the `hpu` device in `PyTorch`. Therefore we will pull the latest docker image with torch installed and run `python3` with the code snippet below. A more detailed guide can be found in [Porting a Simple PyTorch Model to Gaudi](https://docs.habana.ai/en/latest/PyTorch/Migration_Guide/Porting_Simple_PyTorch_Model_to_Gaudi.html).

start docker container with torch installed
```bash
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host vault.habana.ai/gaudi-docker/1.4.1/ubuntu20.04/habanalabs/pytorch-installer-1.10.2:1.4.1-11
```

Start a python3 session with `python3` and execute the code below

```python
import torch_hpu
print(f"device available:{torch_hpu.is_available()}")
print(f"device_count:{torch_hpu.device_count()}")
```

Thats it now you can start using Habana for running your Deep Learning Workload. You can find an example for `text-classification` with `transformers` at the [optimum-habana](https://github.com/huggingface/optimum-habana/tree/main/examples/text-classification) respository.

_P.S. you can also use the `start_instance.sh` script which does all of the steps above._

**clean up**

terminate the ec2 instance
```Bash
INSTANCE_NAME=habana
aws ec2 terminate-instances --instance-ids $(aws ec2 describe-instances --filters "Name=tag:Name,Values=${INSTANCE_NAME}-demo" --query 'Reservations[*].Instances[*].InstanceId' --output text) \
2>&1 > /dev/null
```

security group. _can be delete once the instance is terminated_
```bash
SG_NAME=habana
aws ec2 delete-security-group --group-name ${KEY_NAME}-sg
```

key pair  _can be delete once the instance is terminated_
```bash
KEY_NAME=habana
aws ec2 delete-key-pair --key-name ${KEY_NAME}
rm ${KEY_NAME}.pem
```


