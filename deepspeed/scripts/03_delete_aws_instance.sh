NAME=habana
INSTANCE_ID=$(aws ec2 describe-instances --filters "Name=tag:Name,Values=${NAME}-demo" "Name=instance-state-name,Values=running" --query 'Reservations[*].Instances[*].InstanceId' --output text)

# delete local keypair
echo "deleting local keypair"
rm -f ${NAME}.pem

# terminate instance
echo "terminating instance: ${INSTANCE_ID}"
aws ec2 terminate-instances --instance-ids ${INSTANCE_ID} 2>&1 > /dev/null

# wait for termination
echo "waiting for instance to terminate"
aws ec2 wait instance-terminated --instance-ids ${INSTANCE_ID} 2>&1 > /dev/null

# delte security group
echo "deleting security group"
aws ec2 delete-security-group --group-name ${NAME}-sg 2>&1 > /dev/null

# delete keypair
echo "deleting ec2 keypair"
aws ec2 delete-key-pair --key-name ${NAME}  2>&1 > /dev/null
