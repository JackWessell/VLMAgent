import pandas as pd
import numpy as np
import torch 
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig
import pickle
import sys
import argparse
import gym
from utils import make_conversation
from PIL import Image

def run_hf(environment_name):
    #quantization: not necessary but can improve inference speed at the cost of some performance
    '''for i in range(torch.cuda.device_count()):
           print(torch.cuda.get_device_properties(i).name)
    torch.cuda.set_device(1)
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )'''
    model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", torch_dtype=torch.float16, device_map="auto")
    model.eval()
    #print(model.device)
    processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")
    res = []
    demo_environment = gym.make(environment_name, render_mode='rgb_array')
    state, info = demo_environment.reset()
    for i in range(35):
        action = np.random.randint(low = 0, high = 4)
        state, _, _, _, _ = demo_environment.step(action)
    image = demo_environment.render()
    im = Image.fromarray(image)
    im.save("../agent_outputs/your_file.jpeg")
    eg = Image.open("../agent_outputs/eg.jpeg") 
    eg = np.array(eg)
    images = [eg, image]
    chat = make_conversation(1, ["Fire right."])
    prompt = processor.apply_chat_template(chat, add_generation_prompt=True)
    inputs = processor(images=images, text=prompt, return_tensors="pt").to("cuda:0", torch.float16)
    output = model.generate(**inputs, max_new_tokens=5)
    instruction = processor.decode(output[0], skip_special_tokens=True)
    print(instruction[len(inputs):])
    with open("../agent_outputs/Output.txt", "w") as text_file:
        text_file.write(instruction[len(inputs):])
    return
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--env',type=str, default = 'LunarLander-v2', help='Which environment to use')
    args = parser.parse_args()
    run_hf(args.env)