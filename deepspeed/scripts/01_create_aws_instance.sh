NAME=habana
INSTANCE_TYPE=dl1.24xlarge

AMI_ID=$(aws ec2 describe-images --filters "Name=name,Values=optimum-habana-synapse-1.6.0"  --query 'Images[0].ImageId' --output text)
echo "AMI ID: ${AMI_ID}"

export AWS_PROFILE=hf-sm 
export AWS_DEFAULT_REGION=us-east-1 

# create key pair for ssh
aws ec2 create-key-pair --key-name habana --query 'KeyMaterial' --output text > ${NAME}.pem
chmod 400 ${NAME}.pem

# create security group
DEFAULT_VPC_ID=$(aws ec2 describe-vpcs --query 'Vpcs[?isDefault==true].VpcId' --output text)
echo "Default VPC ID: ${DEFAULT_VPC_ID}"
SECURITY_GROUP_ID=$(aws ec2 create-security-group --group-name ${NAME}-sg --description "SG for Habana Deep Learning" --vpc-id ${DEFAULT_VPC_ID} --output text)
echo "Security Group ID: ${SECURITY_GROUP_ID}"
echo $(aws ec2 authorize-security-group-ingress --group-id ${SECURITY_GROUP_ID} --protocol tcp --port 22 --cidr 0.0.0.0/0 --output text)

# start instance
aws ec2 run-instances \
  --image-id ${AMI_ID} \
  --key-name ${NAME} \
  --count 1 \
  --instance-type ${INSTANCE_TYPE} \
  --security-group-ids ${SECURITY_GROUP_ID} \
  --block-device-mappings 'DeviceName=/dev/sda1,Ebs={VolumeSize=150}' \
  --tag-specifications "ResourceType=instance,Tags=[{Key=Name,Value=${NAME}-demo}]" \
  2>&1 > /dev/null

# wait for instance to start
sleep 5
INSTANCE_ID=$(aws ec2 describe-instances --filters "Name=tag:Name,Values=${NAME}-demo" "Name=instance-state-name,Values=running" --query 'Reservations[*].Instances[*].InstanceId' --output text)
echo "Waiting for the instance ${INSTANCE_ID} to start..."
aws ec2 wait instance-running  --instance-ids ${INSTANCE_ID} 2>&1 > /dev/null
sleep 15
echo "Instance started"
