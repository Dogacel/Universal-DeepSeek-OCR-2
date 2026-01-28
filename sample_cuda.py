from transformers import AutoModel, AutoTokenizer
import torch

model_name = 'Dogacel/Universal-DeepSeek-OCR-2'

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True)
model = model.eval().to("cuda").to(torch.bfloat16)

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
    device = "cuda",
    dtype = torch.bfloat16,
)