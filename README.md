# Gemma 2B English Quotes Fine-Tuning

Fine-tuning Google's Gemma 2B language model on English quotes using **LoRA** and supervised fine-tuning.

The goal is to teach the model to generate quote-style text and continue quote prompts.

## ✨ Highlights

* Fine-tuned Gemma 2B using LoRA (parameter-efficient fine-tuning).
* Trained on approximately 2,508 English quotes.
* Training completed on an NVIDIA Tesla T4 GPU.
* Published the trained LoRA adapter on Hugging Face.

## 🔗 Resources

| Resource         | Link                                                                                 |
| ---------------- | ------------------------------------------------------------------------------------ |
| Base Model       | [Google Gemma 2B](https://huggingface.co/google/gemma-2b)                            |
| Fine-Tuned Model | [Gemma 2B English Quotes](https://huggingface.co/deepak2310/gemma-2b-english-quotes) |

## 🛠️ Tech Stack

Python • PyTorch • Transformers • PEFT • TRL • LoRA • Google Colab

## 📊 Training Configuration

| Parameter             | Value            |
| --------------------- | ---------------- |
| Base model            | google/gemma-2b  |
| Fine-tuning           | LoRA             |
| Trainer               | SFTTrainer       |
| Training steps        | 100              |
| Learning rate         | 2e-4             |
| Batch size            | 1                |
| Gradient accumulation | 4                |
| Optimizer             | paged_adamw_8bit |
| Final training loss   | ~2.13            |
| GPU                   | NVIDIA Tesla T4  |

## 📁 Project Structure

```text
gemma-2b-english-quotes/
│
├── fine_tuning.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## 🚀 Getting Started

### Installation

```bash
pip install transformers peft torch accelerate
```

### Load the Fine-Tuned Model

The Hugging Face repository contains the LoRA adapter. The base Gemma model is loaded separately.

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model = "google/gemma-2b"
adapter_model = "deepak2310/gemma-2b-english-quotes"

tokenizer = AutoTokenizer.from_pretrained(base_model)

model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=torch.float32,
    device_map="auto"
)

model = PeftModel.from_pretrained(
    model,
    adapter_model
)

model.eval()

prompt = "Quote: Be who you are and say what you feel,"

inputs = tokenizer(
    prompt,
    return_tensors="pt"
).to(model.device)

with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=50,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

print(tokenizer.decode(
    outputs[0],
    skip_special_tokens=True
))
```

## 📈 Results

The model was tested on quote prompts and generated text continuations.

* Training steps: 100
* Final training loss: approximately 2.13
* Evaluation: Preliminary testing; no formal held-out test set evaluation.

## ⚠️ Limitations

* May generate fictional or inaccurate quotes.
* Author attributions may be incorrect.
* Outputs may be repetitive.
* Not comprehensively benchmarked.

## 📜 License

The project code is released under the **MIT License**.

The Gemma model and its derivatives are subject to [Google's Gemma Terms of Use](https://ai.google.dev/gemma/terms).

## 👨‍💻 Author

**Deepak Singh**

* GitHub: [DEEPAKSINGH275](https://github.com/DEEPAKSINGH275)
* Hugging Face: [deepak2310](https://huggingface.co/deepak2310)

---
