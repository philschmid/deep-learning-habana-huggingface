# BERT Inference with Hugging Face Transformers on Habana Gaudi

This repository contains instructions for getting started with Deep Learning on Habana Gaudi with Hugging Face libraries like [transformers](https://huggingface.co/docs/transformers/index), [optimum](https://huggingface.co/docs/optimum/index), [datasets](https://huggingface.co/docs/datasets/index). This guide will show you how to use Habana Gaudi for inference workloads with transforemrs like BERT.

You can take a look at [philschmid.de/getting-started-habana-gaudi](https://www.philschmid.de/getting-started-habana-gaudi) on how to setup your environment. The examples are expected to run on a Habana Gaudi instance.

### Examples

* [bert-inference](bert-inference.ipynb)


### Run Dev environment

1. start container

```bash
docker run -it --runtime=habana -e HABANA_VISIBLE_DEVICES=all -e OMPI_MCA_btl_vader_single_copy_mechanism=none --cap-add=sys_nice --net=host --ipc=host -v /home/ubuntu:/home/ubuntu -w /home/ubuntu vault.habana.ai/gaudi-docker/1.6.0/ubuntu20.04/habanalabs/pytorch-installer-1.12.0:latest
```

2. install dependencies

```bash
pip install jupyter optimum[habana]
```

3. start notebook enviromnment

```bash
jupyter notebook --allow-root
```

4. run inference examples