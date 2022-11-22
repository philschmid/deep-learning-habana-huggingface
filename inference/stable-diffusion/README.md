# Stable Diffusion on Habana Gaudi

This repository contains the code (application & infrastructure) to run the Stable Diffusion model on Habana Gaudi devices using AWS Cloud an DL1 Instances.

### Prerequisites

1. AWS Account
2. Install AWS CDK

## Deploy

```bash
cdk bootstrap
```

```bash
cdk deploy
```

# Welcome to your CDK TypeScript project

This is a blank project for CDK development with TypeScript.

The `cdk.json` file tells the CDK Toolkit how to execute your app.

## Useful commands

* `npm run build`   compile typescript to js
* `npm run watch`   watch for changes and compile
* `npm run test`    perform the jest unit tests
* `cdk deploy`      deploy this stack to your default AWS account/region
* `cdk diff`        compare deployed stack with current state
* `cdk synth`       emits the synthesized CloudFormation template


docker run -ti --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host -v $(pwd):/home/ubuntu/dev --workdir=/home/ubuntu/dev vault.habana.ai/gaudi-docker/1.7.0/ubuntu20.04/habanalabs/pytorch-installer-1.12.0:latest 

python3 -m pip install -r app/requirements.txt