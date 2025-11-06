---
title: "potichat (nanochat) - lets remake chatgpt ?!"
date: "2025-16-10"
theme: "Coding,ML"
summary: "making a chatgpt like from scratch to naviguate the whole stack"
image: "https://ph-files.imgix.net/36810de6-302c-443a-90f3-763a9757fc01.png?auto=format&fit=crop"
status: "in progress"
---

Inspired by karapathy's awesome blog post about nanochat, and then followed by the ppl from hugging face.
I will try to train an LLM from scratch, to better understand the ins and outs.
I will be following HF's free course at first [you can find it here](https://huggingface.co/spaces/HuggingFaceTB/smol-training-playbook#super-power-speed-and-data)

# Starting

## The architecture type.

I've chosen to take the same baseline as google's Gemma or Alibaba's Qwen or even HF's SmolLM.
The architecture baseline will be _Dense_. I've Chosen to go with this one rather than an MOE or hybrid because of the final size and parameters of the model.
I still would like to avoid training a 400B model, because it will take too much time, and therefore would be too costy for an experiment (and because I'm impatient :p)
So Let's aim for a >10B model.

## The training framework

In the same spirit as before, I've chosen to go with **TorchTitan** ([here]([https://github.com/pytorch/torchtitan)) because while It was only tested by the pytorch team and is relatively new, It is optimized for Dense type.
It is also much lighter, and therefore (I hope) will take me less time to go through if needed.

_WIP_
