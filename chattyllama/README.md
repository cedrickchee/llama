# ChattyLLaMA

ChattyLLaMA is LLaMA-based ChatGPT.

FAIR is aware [LLaMA generations are unexpected](../FAQ.md#2-generations-are-bad).

This is due to the fact that LLaMA was not trained on conversational prompts.

Here's what they suggest:

> To be able to directly prompt the models with questions / instructions, you can either:
>
> - Prompt it with few-shot examples so that the model understands the task you have in mind.
> - Finetune the models on datasets of instructions to make them more robust to input prompts.

My ideas is to finetune the models on a diverse set of instructions datasets
from LAION's OpenAssistant.

You can finetune language models from human preferences (i.e., Reinforcement
Learning from Human Feedback (RLHF)).

People under-appreciate fine-tuning alone compared to RLHF: many papers show how
far you can get with instruction tuning and no Reinforecement Learning (RL). RL
algorithms are quite finicky — sensitive to picking hard-to-tune hyperparams —
compared to supervised deep learning.

LLaMA paper touches on finetuning briefly, referencing the fine-tuning protocol
from Flan.

[ChatLLaMA](https://github.com/nebuly-ai/nebullvm/tree/main/apps/accelerate/chatllama)
enables you to build a ChatGPT-style service based on pre-trained LLaMA models.

This allows you to train LLaMA-based architectures in a similar way to ChatGPT,
using RLHF.

(_**Disclaimer:** The work is for research purposes._)

## Plan

### LLaMA model weights

_The below is my experiments with all the compression and acceleration techniques, tricks, algorithms, and
more — documented in my [awesome-ml-model-compression](https://github.com/cedrickchee/awesome-ml-model-compression) project._

**Goals:**

- **Inference efficiency:** make models smaller and faster
- **Unlock on-device deployment:** run on low-resources hardwares and consumer GPUs instead of Cloud

Compression:
A classical example is quantization, which compress the weight matrices of a layer, by
reducing its precision (i.e., from 32-bit floating point values to 8-bit unsigned integers), with
minimal loss in quality.

Current high-level plan (tentatively):

- **Weight quantization**
  - Squeeze LLaMA models using 8-bit optimizers and quantization like [Bitsandbyte](https://arxiv.org/abs/2110.02861), [LLM.int8(): 8-bit Matrix Multiplication for Transformers at Scale](https://arxiv.org/abs/2208.07339) [[LLM.int8 blog post](https://huggingface.co/blog/hf-bitsandbytes-integration)].
  - [SmoothQuant: Accurate and Efficient Post-Training Quantization for LLMs](https://arxiv.org/abs/2211.10438) [[code](https://github.com/mit-han-lab/smoothquant)]
  - [ZeroQuant: Efficient and Affordable Post-training Quantization for Large Transformer-based Models](https://arxiv.org/abs/2206.01861)
  - [nuQmm: Quantized MatMul for Efficient Inference of Large-Scale Generative Language Models](https://arxiv.org/abs/2206.09557)
  - [MKQ-BERT: Quantized BERT with 4-bits Weights and Activations](https://arxiv.org/abs/2203.13483)
- **Activation quantization** for latency improvements.
- **Other compression techniques** like low-rank matrix factorization (i.e., [LoRA: Low-Rank Adaptation of LLMs (adapters available for GPT-like models)](https://arxiv.org/abs/2106.09685 )), weight-sharing, K-means clustering.
- **Pruning**
  - Pruning so that the neural network can be made sparse.
  - SparseGPT: [Massive Language Models Can Be Accurately Pruned in One-Shot](https://arxiv.org/abs/2301.00774)
- **Infrastructure**
  
  To run inference efficiently, there has to be a robust software and hardware infrastructure foundation. Examples: XLA for server-side acceleration, DeepSpeed, etc.

  **Offloading:**
  
  Systems that are specialized for LLM inference, such as FasterTransformer (NVIDIA, 2022), PaLM inference (Pope et al., 2022), Deepspeed-Inference (Aminabadi et al., 2022), Accelerate (HuggingFace, 2022), LightSeq (Wang et al., 2021), TurboTransformers (Fang et al., 2021).
  
  To enable LLM inference on easily accessible hardware, offloading is an essential technique — to
  our knowledge, among current systems, only Deepspeed-Inference, Huggingface Accelerate, and [FlexGen](https://raw.githubusercontent.com/FMInference/FlexGen/main/docs/paper.pdf) include such functionality.

  FlexGen claims (unverified) they enable high-throughput inference of LLMs with very a few GPUs.

  **Compilers**

  PyTorch also offers the JIT compilation facility which might seem similar to XLA, but it's not. The alternative for XLA in the PyTorch ecosystem seem to be the Glow and TensorComprehension.
  
  PyTorch offers a [model tuning guide](https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html). 
  
  Hardware-optimized libraries: [XNNPACK](https://github.com/google/XNNPACK) supports several common ops in quantized inference mode for PyTorch.

  **Parallelism, GPU clusters, distributed GPUs environment, Cloud**

  - Model Parallel (MP) encompasses both Pipeline Parallel (PP) and Tensor Parallel (TP)

    Distributed model weights - write a script for resharding MP parts to run the larger model (33B) across on 4 GPUs.
    The script used to reshard OPT from metaseq doesn't work.

    Note about the confusing concepts:

    - PP: shard layers - split the model up vertically (layer-level) across multiple GPUs.
    - TP: split each tensor up into multiple shards, so instead of having the whole tensor reside on a single GPU, each shard of the tensor resides on its designated GPU.

  Port LLaMA PyTorch model (based on metaseq, fairscale?) to HuggingFace Transformers.

To round up, start with [🤗 PEFT: Parameter-Efficient Fine-Tuning of Billion-Scale Models on Low-Resource Hardware](https://huggingface.co/blog/peft) - The HugingFace PEFT library enables using the most popular and performant models from Transformers coupled with the simplicity and scalability of Accelerate. Currently supported PEFT methods: LoRA, prefix tuning, prompt tuning, and P-Tuning (which employs trainable continuous prompt embeddings). They'll be exploring more PEFT methods, such as (IA)3 and bottleneck adapters. Results: The number of parameters needed to fine-tune Flan-T5-XXL is now 9.4M, about 7X fewer than AlexNet.

Future plan:

- Try Bellard's [NNCP](https://bellard.org/nncp/) - a practical lossless data compressor with neural networks. The latest version uses a Transformer model (slower but best ratio). LSTM (faster) is also available.
- MLOps: _[Breakdown of Nvidia H100s for Transformer Inferencing](https://carolchen.me/blog/h100-inferencing/)_.
- Reading: [Understanding and Overcoming the Challenges of Efficient Transformer Quantization](https://arxiv.org/abs/2109.12948)
- Reading: [The case for 4-bit precision: k-bit Inference Scaling Laws](https://arxiv.org/abs/2212.09720) by Tim Dettmers et al. (2022) - Overall, their findings show that **4-bit precision is almost universally optimal for total model bits and zero-shot accuracy**.
- Distilling model down (knowledge transfer) to a smaller model and sharing the student model as open source?
- Extend LLaMA model: adapt it so that we can pass-in a new sampler and more controls for generations, like repetition penalty. Some [code snippets](https://rentry.org/llama_few_more_samplers).

### ChattyLLaMA

(TODO)
