# Fine-Tuning Hugging Face Transformers with Optimum on Habana Gaudi1 using distributed training


In this blog, you will learn how to fine-tune BERT for multi-class text-classification using a Habana Gaudi instance on AWS to save 40% cost and be 2x faster then comparable GPUs.
We will use the Hugging Faces Transformers, Optimum and Datasets library to fine-tune a pre-trained transformer for multi-class text classification. In particular, we will fine-tune BERT-Large using the Banking77 dataset. Before we get started, we need to set up the deep learning environment. 

You will learn how to:

1. Setup Habana Gaudi instance
2. Load and process the dataset
3. Create a `GaudiTrainer` and define the training arguments

### Requirements

Before we can start make sure you have met the following requirements

* AWS Account with quota for [DL1 instance type](https://aws.amazon.com/ec2/instance-types/dl1/)
* [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html) installed
* AWS IAM user [configured in CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html) with permission to create and manage ec2 instances

### Helpful Resources

* [Optimum Habana Documentation](https://github.com/huggingface/optimum-habana)

## 1. Setup Habana Gaudi instance

In this example are we going to use Habana Gaudi on AWS using the DL1 instance. We already have created a blog post in the past on how to [Setup Deep Learning environment for Hugging Face Transformers with Habana Gaudi on AWS](https://www.philschmid.de/getting-started-habana-gaudi). If you haven't have read this blog post, please read it first and go through the steps on how to setup the environment. 
Or if you feel comfortable you can use the `start_instance.sh` in the root of the repository to create your DL1 instance and the continue at step  [4. Use Jupyter Notebook/Lab via ssh](https://www.philschmid.de/getting-started-habana-gaudi#4-use-jupyter-notebooklab-via-ssh) in the Setup blog post.

1. run habana docker container an mount current directory
```bash
docker run -ti --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host -v $(pwd):/home/ubuntu/dev --workdir=/home/ubuntu/dev vault.habana.ai/gaudi-docker/1.4.1/ubuntu20.04/habanalabs/pytorch-installer-1.10.2:1.4.1-11
```
2. install juptyer
```bash
pip install jupyter
```

3. clone repository
```bash
git clone https://github.com/philschmid/deep-learning-habana-huggingface.git
cd fine-tuning
```

4. run jupyter notebook
```bash
jupyter notebook --allow-root
#         http://localhost:8888/?token=f8d00db29a6adc03023413b7f234d110fe0d24972d7ae65e
```
4. continue here

_**NOTE**: The following steps assume that the code/cells are running on a gaudi instance with access to HPUs_


# Results

Below you can find the results of the fine-tuning `XLM-RoBERTa-large` on the `MASSIVE` dataset using Habana Gaudi and NVIDIA V100.

| accelerator        | training time (in minutes) | total cost | total batch size | aws instance type                                                    | instance price per hour |
|--------------------|----------------------------|------------|------------------|----------------------------------------------------------------------|-------------------------|
| Habana Gaudi (HPU) | 52.6                       | $11,55     | 64               | [dl1.24xlarge](https://aws.amazon.com/ec2/instance-types/dl1/)       | $13.11                  |
| NVIDIA V100 (GPU)  |                            |            |                  | [p3.8xlarge](https://aws.amazon.com/ec2/instance-types/p3/?nc1=h_ls) | $12.24                  |


## Habana Gaudi [dl1.24xlarge](https://aws.amazon.com/ec2/instance-types/dl1/)

**train results**

```bash
{'loss': 0.2651, 'learning_rate': 2.4e-05, 'epoch': 1.0}
{'loss': 0.1079, 'learning_rate': 1.8e-05, 'epoch': 2.0}
{'loss': 0.0563, 'learning_rate': 1.2e-05, 'epoch': 3.0}
{'loss': 0.0308, 'learning_rate': 6e-06, 'epoch': 4.0}
{'loss': 0.0165, 'learning_rate': 0.0, 'epoch': 5.0}
```

total
```bash
{'train_runtime': 3172.4502, 'train_samples_per_second': 127.028, 'train_steps_per_second': 1.986, 'train_loss': 0.09531746031746031, 'epoch': 5.0}
```


**eval results**

```bash
{'eval_loss': 0.3128528892993927, 'eval_accuracy': 0.9125852013210597, 'eval_f1': 0.9125852013210597, 'eval_runtime': 45.1795, 'eval_samples_per_second': 314.988, 'eval_steps_per_second': 4.936, 'epoch': 1.0}
{'eval_loss': 0.36222779750823975, 'eval_accuracy': 0.9134987000210807, 'eval_f1': 0.9134987000210807, 'eval_runtime': 29.8241, 'eval_samples_per_second': 477.165, 'eval_steps_per_second': 7.477, 'epoch': 2.0}
{'eval_loss': 0.3943144679069519, 'eval_accuracy': 0.9140608530672476, 'eval_f1': 0.9140
608530672476, 'eval_runtime': 30.1085, 'eval_samples_per_second': 472.657, 'eval_steps_per_second': 7.407, 'epoch': 3.0}
{'eval_loss': 0.40938863158226013, 'eval_accuracy': 0.9158878504672897, 'eval_f1': 0.9158878504672897, 'eval_runtime': 30.4546, 'eval_samples_per_second': 467.286, 'eval_steps_per_second': 7.322, 'epoch': 4.0}
{'eval_loss': 0.4137658476829529, 'eval_accuracy': 0.9172932330827067, 'eval_f1': 0.9172932330827067, 'eval_runtime': 30.3464, 'eval_samples_per_second': 468.952, 'eval_steps_per_second': 7.348, 'epoch': 5.0}
```

## NVIDIA V100 [p3.8xlarge](https://aws.amazon.com/ec2/instance-types/p3/?nc1=h_ls)

