import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def simple_chat_demo():
    # 模型路径
    model_path = r"C:\Users\PC\.cache\modelscope\hub\models\deepseek-ai\DeepSeek-R1-Distill-Qwen-1.5B"

    print("正在加载模型和tokenizer...")

    try:
        # 加载tokenizer和模型
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True
        )

        print("✅ 模型加载成功！")
        print(f"模型设备: {model.device}")

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    print("\n🤖 开始对话（输入 'quit' 退出）")
    print("=" * 50)

    while True:
        user_input = input("\n👤 您: ").strip()

        if user_input.lower() in ['quit', 'exit', '退出']:
            print("再见！👋")
            break

        if not user_input:
            continue

        try:
            # 简单构建prompt
            prompt = f"用户: {user_input}\n助手:"

            # 编码输入
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

            # 生成回复
            outputs = model.generate(
                **inputs,
                max_new_tokens=256,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )

            # 解码回复
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # 只提取助手回复部分
            response = response.split("助手:")[-1].strip()

            print(f"🤖 助手: {response}")

        except Exception as e:
            print(f"❌ 生成回复时出错: {e}")


if __name__ == "__main__":
    simple_chat_demo()