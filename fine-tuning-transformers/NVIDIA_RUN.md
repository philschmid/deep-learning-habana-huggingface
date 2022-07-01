# Steps to run the script on NVIDIA V100:

## 1. create and ssh into instance.

```bash
export AWS_PROFILE=sandbox
export AWS_DEFAULT_REGION=us-east-1
```

```bash
KEY_NAME=nvidia

# create key pair for ssh
aws ec2 create-key-pair --key-name ${KEY_NAME} --query 'KeyMaterial' --output text > ${KEY_NAME}.pem
chmod 400 ${KEY_NAME}.pem

# create security group
SG_NAME=nvidia
DEFAULT_VPC_ID=$(aws ec2 describe-vpcs --query 'Vpcs[?isDefault==true].VpcId' --output text)
echo "Default VPC ID: ${DEFAULT_VPC_ID}"
SECURITY_GROUP_ID=$(aws ec2 create-security-group --group-name ${SG_NAME}-sg --description "SG for nvidia Deep Learning" --vpc-id ${DEFAULT_VPC_ID} --output text)
echo "Security Group ID: ${SECURITY_GROUP_ID}"
echo $(aws ec2 authorize-security-group-ingress --group-id ${SECURITY_GROUP_ID} --protocol tcp --port 22 --cidr 0.0.0.0/0 --output text)

# start instance
AMI_ID=ami-02ad71b12f6c813ab
echo "AMI ID: ${AMI_ID}"
INSTANCE_TYPE=p3.8xlarge
INSTANCE_NAME=nvidia

aws ec2 run-instances \
  --image-id ${AMI_ID} \
  --key-name ${KEY_NAME} \
  --count 1 \
  --instance-type ${INSTANCE_TYPE} \
  --security-group-ids ${SECURITY_GROUP_ID} \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150}' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${INSTANCE_NAME}-demo}]" \
  2>&1 > /dev/null

# connect via ssh
echo "waiting for the instance to start..."
sleep 45

PUBLIC_DOMAIN=$(aws ec2 describe-instances \
    --filters Name=tag-value,Values=nvidia-demo  \
    --query 'Reservations[*].Instances[*].PublicDnsName' \
    --output text)

echo "connect to instance via ssh with:\nssh -L 8888:localhost:8888 -i ${KEY_NAME}.pem ubuntu@${PUBLIC_DOMAIN//[$'\t\r\n ']}"
```

example
```bash
ssh -L 8888:localhost:8888 -i nvidia.pem ubuntu@ec2-34-235-163-218.compute-1.amazonaws.com
```


## run training

```bash
git clone https://github.com/philschmid/deep-learning-habana-huggingface.git
cd deep-learning-habana-huggingface
git checkout fine-tuning
cd fine-tuning-transformers

```

check cuda 


```bash
docker run -it --runtime=nvidia --cap-add=sys_nice --net=host --ipc=host -v /home/ubuntu:/home/ubuntu -w /home/ubuntu  anibali/pytorch:1.10.2-cuda11.3 nvidia-smi
```

start container

```bash
docker run -it --runtime=nvidia --cap-add=sys_nice --net=host --ipc=host -v /home/ubuntu/deep-learning-habana-huggingface/fine-tuning-transformers:/home/ubuntu -w /home/ubuntu  anibali/pytorch:1.10.2-cuda11.3 /bin/bash
```

install dependencies

```bash
conda init && exec bash && conda activate pytorch \
&& sudo apt update && sudo apt install git-lfs -y \
&& pip install transformers datasets sklearn tensorboard
```

log into huggingface
```bash
huggingface-cli login
```
or save token directly 

```bash
mkdir ~/.huggingface && echo -n "hf_xx" > ~/.huggingface/token
```

run distributed training
```bash
python3 -m torch.distributed.launch --nproc_per_node=4 scripts/train.py
```

run training
```bash
CUDA_VISIBLE_DEVICES="0" python3  scripts/train.py
```