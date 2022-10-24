NAME=habana
PUBLIC_DOMAIN=$(aws ec2 describe-instances \
    --filters Name=tag-value,Values=habana-demo  \
    --query 'Reservations[*].Instances[*].PublicDnsName' \
    --output text)

PORT=$(( ( RANDOM % 5000 )  + 4000 ))

scp -r -i ${NAME}.pem $(pwd)/src ubuntu@${PUBLIC_DOMAIN//[$'\t\r\n ']}:/home/ubuntu/dev/src
scp -r -i ${NAME}.pem $(pwd)/t5_summarization_z2.ipynb ubuntu@${PUBLIC_DOMAIN//[$'\t\r\n ']}:/home/ubuntu/dev/t5_summarization_z2.ipynb

ssh -L ${PORT}:localhost:${PORT} -i ${NAME}.pem ubuntu@${PUBLIC_DOMAIN//[$'\t\r\n ']} <<ENDSSH
cd dev && jupyter notebook --port ${PORT}
ENDSSH

