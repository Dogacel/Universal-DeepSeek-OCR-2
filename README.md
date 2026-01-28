<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

 # _Universal_ DeepSeek-OCR 2 – CPU, MPS, CUDA Support 


<div align="left">
  <a href="https://huggingface.co/Dogacel/Universal-DeepSeek-OCR-2" target="_blank">
    <img alt="Hugging Face" src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Universal%20DeepSeek%20OCR%202-ffc107?color=ffc107&logoColor=white" />
  </a>
</div>

This repository uses the weights from the original DeepSeek-OCR 2 and modifies model to support inference on different devices such as CPU and MPS (Apple Metal GPU). By default runs on CPU.

- [Link to the original DeepSeek-OCR-2 model](https://huggingface.co/deepseek-ai/DeepSeek-OCR-2)

- [Link to the original DeepSeek-OCR-2 repository](https://github.com/deepseek-ai/DeepSeek-OCR-2)

<hr/>

<div align="center">
  <img src="assets/logo.svg" width="60%" alt="DeepSeek AI" />
</div>




<h2>
  <p align="center">
    DeepSeek-OCR 2: Visual Causal Flow
  </p>
</h2>

<p align="center">
<img src="assets/fig1.png" style="width: 600px" align=center>
</p>
<p align="center">
  Explore more human-like visual encoding. 
</p>


## Usage

Unlike the original _DeepSeek-OCR-2_ repository, **Universal** version works on different device types, such as CPU, MPS and even CUDA.

1. Clone this repository and navigate to the DeepSeek-OCR-2 folder
```bash
git clone https://github.com/Dogacel/Universal-DeepSeek-OCR-2.git
cd Universal-DeepSeek-OCR-2
```

2. Install Dependencies
```Shell
conda create -n deepseek-ocr2 python=3.12.9 -y
conda activate deepseek-ocr2
pip install -r requirements.txt
```

3. Run

Choose the sample to run.

```Shell
python sample_cpu.py  # For CPU inference

export PYTORCH_ENABLE_MPS_FALLBACK=1
python sample_mps.py  # For Apple Metal GPU inference

python sample_cuda.py # For NVIDIA GPU inference
```

**Note:** if you want to use the CUDA, you might need to install torch from a wheel that is built using CUDA using a command such as `pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu118`.


## Sample Code

```python
from transformers import AutoModel, AutoTokenizer
import torch

model_name = 'Dogacel/Universal-DeepSeek-OCR-2'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True)
model = model.eval().to("cpu").to(torch.float16)

# prompt = "<image>\nFree OCR. "
prompt = "<image>\n<|grounding|>Convert the document to markdown. "
image_file = 'sample/paper.png'
output_path = 'output'

res = model.infer(
    tokenizer, 
    prompt=prompt, 
    image_file=image_file, 
    output_path = output_path, 
    base_size = 1024, 
    image_size = 768, 
    crop_mode = True, 
    save_results = True,
    test_compress = True,
)
```

## vLLM-Inference

For vLLM inference support, refer to the [original DeepSeek-OCR-2 repository](https://github.com/deepseek-ai/DeepSeek-OCR-2/tree/main).
