import base64
from io import BytesIO
import traceback
from PIL import Image
import numpy as np
import json
import time
from dashscope import MultiModalConversation

def encode_image(image_files):
    base_img = []
    for file in image_files:
        # 1. 检查并处理 Numpy 数组类型 (来自 Isaac Sim)
        if isinstance(file, np.ndarray):
            # 确保内存连续，并转为 uint8 格式
            file = np.ascontiguousarray(file)
            if file.dtype != np.uint8:
                file = file.astype(np.uint8)
            # 转为 PIL Image
            img = Image.fromarray(file)
        # 2. 如果已经是 PIL Image 类型
        elif isinstance(file, Image.Image):
            img = file
        else:
            raise TypeError(f"不支持的图像数据类型: {type(file)}")

        # 3. 将图像压缩为标准的 PNG 字节流存储在内存中
        buffered = BytesIO()
        img.save(buffered, format="PNG")

        # 4. 对标准图像字节流进行 Base64 编码
        encoded_string = base64.b64encode(buffered.getvalue()).decode("utf-8")
        base_img.append(encoded_string)

    return base_img


def generate_caption(image_file, temperature=0.7):
    n = len(image_file)
    system_prompt = f"""
            You are an image understanding assistant. Your task is to generate one concise and detailed description per image that focuses on:

            1. **Key objects and their core attributes** - List every object with simple nouns **and** one or two attributes (e.g., “yellow slide, medium-sized, plastic”; “Springer Spaniel dog, tricolor coat”; “gas station building, metal canopy”).  

            2. **Object quantities and groupings** - Specify number if more than one or if it’s a cluster (e.g., “three children”, “a row of parked cars”).  

            3. **Precise spatial relationships** - Describe relative positions, distances or directions (e.g., “the dog sits immediately to the right of the slide”, “the building stands in the distant background, slightly left of center”).  

            4. **Object states or actions** - Note any visible activity or condition (e.g., “children sliding down”, “pump nozzles hanging idle”, “car doors open”).  

            5. **Avoid opinions or irrelevancies** - Use plain factual language. Do not include judgments, emotional tone words, or fine-grained internal part details.  

            User will supply {n} images; you must return exactly one well-structured description string for every image(no quotation marks).
            For each image, return exactly only one description item. 
            You are not allowed to return a blank caption string or more than {n} caption for {n} images.
            Return your answer **only** as a JSON array of caption strings (no prose outside the array), like:
            Example output exactly:
            [
                "yellow slide, medium-sized, plastic; slide is in the foreground; children sliding down",
                "......",
                "gas station building, metal canopy; two fuel pumps; pump nozzles hanging idle; canopy in the background"
            ]
            Do not write anything else.
            IMPORTANT INSTRUCTIONS:
            - You must respond **only in valid JSON format**.
            - Do **not** include any extra explanation, commentary, markdown formatting, or newlines.
            - The response must be a **single-line JSON object**, not a stringified object or list.
    """

    captions = []
    user_content = []
    for img_b64 in image_file:
        user_content.append({'image': 'data:image/png;base64,' + img_b64})
    messages = [{"role": "system", "content": [system_prompt]},
                {'role': 'user', 'content': user_content}]

    # ==============================================================
    # 终极保护：任何网络报错、数据丢失，全部接管，绝不让程序崩溃！
    # ==============================================================
    try:
        response = MultiModalConversation.call(
            api_key="sk-528d0d3ce55f4c5f830dc00df4733eb5",
            model='qwen-vl-max',
            messages=messages
        )

        # 检查响应是否为空
        if not response:
            print(">>> [警告] DashScope API 返回为空 (可能是代理拦截)！使用兜底描述。")
            return ["uniform gray background; no objects present"] * n

        # 检查 output 是否存在且不为 None
        if "output" not in response or response["output"] is None:
            print(f">>> [警告] DashScope API 没有 output 数据！使用兜底描述。")
            return ["uniform gray background; no objects present"] * n

        # 安全提取文本内容
        raw = response["output"]["choices"][0]["message"].content[0]["text"].strip()

        # 解析 JSON 格式
        captions = json.loads(raw)

        # 检查数量是否对齐
        if len(captions) != n:
            print(f">>> [警告] 大模型输出的描述数量 ({len(captions)}) 和图片数量 ({n}) 不匹配！使用兜底描述。")
            return ["uniform gray background; no objects present"] * n

        return captions

    except Exception as e:
        # 这个 except 能够抓住所有的 TypeError, ValueError, KeyError 等等！
        print(f">>> [终极拦截] DashScope API 发生异常: {e}，启用兜底机制！")
        return ["uniform gray background; no objects present"] * n