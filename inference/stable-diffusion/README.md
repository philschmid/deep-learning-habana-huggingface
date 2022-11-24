# Stable Diffusion on Habana Gaudi

This repository contains the code (application & infrastructure) to run the Stable Diffusion model on Habana Gaudi devices using AWS Cloud an DL1 Instances.

### Prerequisites

1. AWS Account
2. Install AWS CDK

## Deploy

```bash
HF_HUB_TOKEN=hf_xx cdk bootstrap
```

```bash
HF_HUB_TOKEN=hf_xx cdk deploy
```

## Local Development

**normal**

```bash
HF_HUB_TOKEN=hf_x python3 -m uvicorn app.main:app  --workers 1
```

**container**

```bash
docker build -t habana-sd -f app/container/Dockerfile app/
```

```bash
docker run -ti --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none -e HF_HUB_TOKEN=hf_xx --cap-add=sys_nice --net=host --ipc=host habana-sd
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
