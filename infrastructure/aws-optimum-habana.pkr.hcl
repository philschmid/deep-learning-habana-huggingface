packer {
  required_plugins {
    amazon = {
      version = ">= 1.1.5"
      source  = "github.com/hashicorp/amazon"
    }
  }
}

source "amazon-ebs" "optimum_habana" {
  ami_name      = local.ami_name
  ami_regions   = ["us-east-1", "us-west-2"]
  ami_groups    = ["all"] # all to make public
  instance_type = var.instance_type
  region        = "us-east-1"
  profile       = var.profile

  source_ami_filter {
    filters = {
      name                = "*Deep Learning AMI Habana*PyTorch*1.12.0*SynapseAI*1.6.0*(Ubuntu 20.04)*"
      root-device-type    = "ebs"
      virtualization-type = "hvm"
    }
    most_recent = true
    owners      = ["898082745236"]
  }
  ssh_username = "ubuntu"
}

build {
  sources = [
    "source.amazon-ebs.optimum_habana"
  ]
  provisioner "shell" {
    environment_vars = [
      "TRANSFORMERS_VERSION=4.23.1",
    ]
    inline = [
      "sudo apt-get update",
      "sudo apt-get -y upgrade --only-upgrade systemd openssl cryptsetup",
      "sudo apt-get install -y bzip2 curl git git-lfs tar libsndfile1-dev ffmpeg",
      "python3 -m  pip install tensorboard transformers[sklearn,sentencepiece,audio,vision]==4.23.1 datasets==2.6.1 evaluate==0.3.0 optimum==1.4.0 optimum-habana==1.2.3 rouge-score nltk git+https://github.com/HabanaAI/DeepSpeed.git@1.6.1  --no-cache-dir",
    ]
  }
}