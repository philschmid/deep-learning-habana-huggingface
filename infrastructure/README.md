# Infrastructure documentation

This directory contains the infrastructure for the project. Included are:

* creating custom AMI images for `optimum-habana`



## Creating custom AMI images for `optimum-habana`

1. Install packer: https://learn.hashicorp.com/tutorials/packer/get-started-install-cli?in=packer/aws-get-started
2. init packer: `packer init .`
2. validate: `packer validate .`
3. check if you want to change the variables or base ami, e.g. new synapse version, or aws profile
4. build: `packer build .`
   1. This will create two public AMIs in `us-east-1` and `eu-west-1` regions, with `optimum-habana` and `transformers` installed.
   2. the build takes around 20minutes

```bash
==> Wait completed after 20 minutes 7 seconds

==> Builds finished. The artifacts of successful builds are:
--> amazon-ebs.optimum_habana: AMIs were created:
us-east-1: ami-04706575ca8435e49
us-west-2: ami-0a35b94ee08e56a00
```