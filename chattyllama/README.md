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
