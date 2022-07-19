# Pre-Training BERT with Hugging Face Transformers on Habana Gaudi

In this blog, you will learn how to pre-train BERT from scratch using a Habana Gaudi instance on AWS to save 40% cost and be 2x faster then comparable GPUs.
We will use the Hugging Faces Transformers, Optimum and Datasets library to pre-train BERT using masked-language modelling one of the two original BERT pre-training tasks. 

You will learn how to:

1. Prepare the dataset
2. Train a Tokenizer
3. Preprocess the dataset
4. Setup Habana Gaudi instance
5. Pre-train BERT on Habana Gaudi

### Requirements

Before we can start make sure you have met the following requirements

* AWS Account with quota for [DL1 instance type](https://aws.amazon.com/ec2/instance-types/dl1/)
* [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) installed
* AWS IAM user [configured in CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) with permission to create and manage ec2 instances

### Helpful Resources

* [Setup Deep Learning environment for Hugging Face Transformers with Habana Gaudi on AWS](https://www.philschmid.de/getting-started-habana-gaudi)
* [Optimum Habana Documentation](https://huggingface.co/docs/optimum/main/en/habana_index)
* [Pre-training script](./scripts/run_mlm.py)
