import os, re, csv, json, torch, base64
import random, argparse
import numpy as np
from PIL import Image
from random import seed
from openai import OpenAI
from tqdm.auto import tqdm
from collections import defaultdict
from transformers import AutoModelForCausalLM, AutoProcessor
# Llama-3.2-11B-Vision
from transformers import MllamaForConditionalGeneration
# Qwen2-VL
from transformers import Qwen2VLForConditionalGeneration
# Qwen2.5-VL
from qwen_vl_utils import process_vision_info
from transformers import Qwen2_5_VLForConditionalGeneration
# LlavaOnevision
from transformers import LlavaOnevisionForConditionalGeneration
# Intern2.5/3
from lmdeploy.vl import load_image
from lmdeploy import pipeline, TurbomindEngineConfig, ChatTemplateConfig
# LlavaNextVideo
from transformers import LlavaNextVideoProcessor, LlavaNextVideoForConditionalGeneration


seed(1234)
np.random.seed(1234)

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, default="gpt-4o")
# parser.add_argument("--device", type=int, default=-1)
args = parser.parse_args()

model_path = args.model_path
model_name = model_path.split("/")[-1]
# device = torch.device(f"cuda:{args.device}" if args.device >= 0 else "cpu")
prompt_format = "\nReply only to the corresponding option.\nAnswer:"

# Set the size of the incoming image for qwen
min_pixels = 256*28*28
max_pixels = 1280*28*28


# Set up the model
if model_name == 'gemini-2.0-flash-001':
    API_KEY = ""  # your api key
    base_url = ""  # Change to your own base_url
    client = OpenAI(api_key=API_KEY, base_url=base_url)
    print(f"Model gemini-2.0-flash series:{model_name} is running!")
    
elif model_name == 'gpt-4o':
    client = OpenAI(api_key="")  # your api key
    client.base_url = ""  # Change to your own base_url
    print(f"Model gpt-4o series:{model_name} is running!")

elif "Qwen2.5-VL" in model_name:
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
    print(f"Model Qwen2.5-VL series:{model_name} is running!")

elif "Qwen2-VL" in model_name :
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path, min_pixels=min_pixels, max_pixels=max_pixels)
    print(f"Model Qwen2-VL series:{model_name} is running!")

elif "InternVL2_5" in model_name:
    model = model_path
    pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=1000000))
    print(f"Model InternVL2_5 series:{model_name} is running!")

elif "LLaVA-NeXT" in model_name:
    model = LlavaNextVideoForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    processor = LlavaNextVideoProcessor.from_pretrained(model_path)
    print(f"Model LLaVA-NeXT series:{model_name} is running!")
    
elif model_name == "llava-onevision-qwen2-7b-ov-hf":
    model = LlavaOnevisionForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        device_map="auto"
    )
    processor = AutoProcessor.from_pretrained(model_path)
    print(f"Model llava-onevision series:{model_name} is running!")

elif model_name == "Llama-3.2-11B-Vision-Instruct":
    model = MllamaForConditionalGeneration.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    processor = AutoProcessor.from_pretrained(model_path)
    print(f"Model Llama-3.2-11B-Vision series:{model_name} is running!")
    
elif model_name == "Kimi-VL-A3B-Instruct":
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
    print(f"Model Kimi-VL series:{model_name} is running!")
    
elif model_name == "InternVL3-14B":
    model = model_path
    pipe = pipeline(model, backend_config=TurbomindEngineConfig(session_len=1000000,tp=1), chat_template_config=ChatTemplateConfig(model_name='internvl2_5'))
    print(f"Model InternVL3 series:{model_name} is running!")
elif model_name == "random":
    model = None
    processor = None
else:
    model = None
    processor = None

def extract_option(text):
    match = re.search(r"\b([A-D])\b", text, re.IGNORECASE)
    return match.group(1).upper() if match else None

def url_to_base64(url):
    if os.path.exists(url):
        with open(url, "rb") as f:
            return "data:image/jpeg;base64," + base64.b64encode(f.read()).decode("utf-8")
    else:
        print(f"该图片{url}不存在！")
        return False



def get_output(image_path, question):
    image_url = [url_to_base64(image) for image in image_path]

    if model_name == 'gemini-2.0-flash-001':
        content = [{"type": "image_url", "image_url": {"url": path}} for path in image_url]

        chat_completion = client.chat.completions.create(
            model='google/gemini-2.0-flash-001',
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        *content
                    ]
                }
            ]
        )
        pred = chat_completion.choices[0].message.content

    elif model_name == 'gpt-4o':
        content = [{"type": "image_url", "image_url": {"url": path}} for path in image_url]

        chat_completion = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": question
                        },
                        *content
                    ]
                }
            ]
        )
        pred = chat_completion.choices[0].message.content


    elif "Qwen2.5-VL" in model_name:
        content = [{"type": "image", "image": path,"resized_height": 280,"resized_width": 420} for path in image_path]

        messages = [
            {
                "role": "user",
                "content": [
                    *content,
                    {
                        "type": "text",
                        "text": question
                    },
                ],
            }
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        )
        inputs = inputs.to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=128)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        pred = str(output_text[0])

    elif "LLaVA-NeXT" in model_name:
        content = [{"type": "image_url", "image_url": {"url": path}} for path in image_url]

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    *content
                ],
            },
        ]
        inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, padding=True, return_tensors="pt").to("cuda")
        generate_ids = model.generate(**inputs, max_new_tokens=100, eos_token_id=2, pad_token_id=2)
        pred = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        match = re.search(r'ASSISTANT:\s*(.*)', pred, re.DOTALL)
        pred = match.group(1)
            
    elif "InternVL2_5" in model_name:
        images = [load_image(image) for image in image_url]
        formatted_lines = ''
        for i, item in enumerate(images, start=1):
            formatted_lines = formatted_lines + "Image-" + str(i) + ": {IMAGE_TOKEN}\n"
        response = pipe((f'{formatted_lines}{question}', images))
        pred = response.text
            
    elif model_name == "llava-onevision-qwen2-7b-ov-hf":
        content = [{"type": "image_url", "image_url": {"url": path}} for path in image_url]

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    *content
                ],
            },
        ]
        inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, padding=True, return_tensors="pt").to("cuda")
        generate_ids = model.generate(**inputs, max_new_tokens=100)
        pred = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]

            
    elif model_name == "Llama-3.2-11B-Vision-Instruct":
        content = [{"type": "image_url", "image_url": {"url": path}} for path in image_url]

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": question},
                    *content
                ],
            },
        ]
        inputs = processor.apply_chat_template(conversation, add_generation_prompt=True, tokenize=True,
                                               return_dict=True, padding=True, return_tensors="pt").to("cuda")
        generate_ids = model.generate(**inputs, max_new_tokens=100)
        pred = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

    elif model_name == "Kimi-VL-A3B-Instruct":
        images_ = [Image.open(path) for path in image_path]
        images = [path.resize((path.width // 4, path.height // 4), Image.Resampling.LANCZOS) for path in images_]
        content = [{"type": "image", "image": path} for path in images]
        messages = [
            {
                "role": "user",
                "content": [ {"type": "text","text": question},*content]
            },
        ]
        text = processor.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt")
        inputs = processor(images=images, text=text, return_tensors="pt", padding=True, truncation=True).to("cuda")
        generated_ids = model.generate(**inputs, max_new_tokens=2048)
        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
        ]
        pred = processor.batch_decode(
            generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

    elif model_name == "InternVL3-14B":
        images = [load_image(image) for image in image_url]
        formatted_lines = ''
        for i, item in enumerate(images, start=1):
            formatted_lines = formatted_lines + "Image-" + str(i) + ": {IMAGE_TOKEN}\n"
        response = pipe((f'{formatted_lines}{question}', images))
        pred = response.text

    else:
        pred = ''
        
    return pred

def evaluate_vlm(benchmark_file):
    with open(benchmark_file, "r", encoding="utf-8") as f:
        benchmark_data = json.load(f)

    stats = defaultdict(lambda: {"correct": 0, "total": 0})
    total_correct = 0
    total_questions = 0

    output_path = f"result/{model_name}"
    if not os.path.exists(output_path):
        os.makedirs(output_path)

    result_file = f"{output_path}/result_{model_name}.csv"
    with open(result_file, "w", newline="", encoding="utf-8") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(["ID", "Question", "Question_Type", "Predicted Answer", "Correct Answer", "IsCorrect"])

        for i, item in enumerate(tqdm(benchmark_data)):
            try:
                image_path = item['image_path']
                question = item["question"] + prompt_format
                correct_answer = item["answer"]
                question_type = item["question_type"]
                stats[question_type]["total"] += 1
                total_questions += 1
                
                predicted_answer = get_output(image_path, question)
                predicted_answer_ = predicted_answer.split("\n")[-1]
                is_correct = extract_option(predicted_answer_) == extract_option(correct_answer)
                
                if is_correct:
                    stats[question_type]["correct"] += 1
                    total_correct += 1
                writer.writerow([i, question, question_type, predicted_answer, correct_answer, is_correct])
            except Exception as e:
                print(f"Error on item {i}: {e}")
                continue

    print("Benchmark Evaluation Results:")
    print("----------------------------------------------------------")
    for qtype, values in stats.items():
        correct = values["correct"]
        total = values["total"]
        accuracy = correct / total
        print(f"{qtype}: {correct}/{total} = {accuracy:.2%}")
    overall_accuracy = total_correct / total_questions
    print("----------------------------------------------------------")
    print(f"The accuracy rate of {model_name} on the benchmark test set: {overall_accuracy:.2%}      Correct quantity:{total_correct}  Total quantity:{total_questions}")
    print("----------------------------------------------------------")
    print(f"The result has been saved to {result_file}")

if __name__ == '__main__':
    benchmark_file = "eval/ViewSpatial-Bench.json"
    evaluate_vlm(benchmark_file)