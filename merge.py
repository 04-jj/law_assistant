from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

# 设置路径
base_model_path = r"C:\Users\PC\.cache\modelscope\hub\models\deepseek-ai\DeepSeek-R1-Distill-Qwen-1.5B"
adapter_path = r"F:\train"
output_path = r"D:\DeepSeek-R1-Distill-Qwen-1.5B-Law"

print("正在加载原模型...")
base_model = AutoModelForCausalLM.from_pretrained(
    base_model_path,
    torch_dtype=torch.float16,  # 使用半精度减少内存占用
    device_map="auto",
    trust_remote_code=True
)

print("正在加载分词器...")
tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)

# 2. 加载适配器
print("正在加载适配器...")
model = PeftModel.from_pretrained(base_model, adapter_path)

# 3. 合并模型
print("正在合并模型...")
merged_model = model.merge_and_unload()

# 4. 保存合并后的模型
print(f"正在保存合并后的模型到 {output_path}...")
merged_model.save_pretrained(output_path)
tokenizer.save_pretrained(output_path)

print("模型合并完成！")