#NOTE: The code below is configured to make running the code possible in Pace-ICE ondemand. You should either change the lines below to reflect your virtual env or comment them out.
import sys
sys.executable = 'miniconda3/envs/DRL/bin/python3.10'
sys.path += ['/home/hice1/jwessell6/DRL/VLMAgent/Gym-Snake', '/home/hice1/jwessell6/miniconda3/envs/DRL/lib/python3.10', 
    '/home/hice1/jwessell6/miniconda3/envs/DRL/lib/python3.10/site-packages', '/home/hice1/jwessell6/miniconda3/envs/DRL/lib/python3.10/lib-dynload', '/home/hice1/jwessell6/.local/bin']

import pandas as pd
import numpy as np
import torch 
from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig
import pickle
import argparse
import gym
import utils 
from PIL import Image

import gym_snake

def run_hf(environment_name):
    #quantization: not necessary but can improve inference speed at the cost of some performance
    '''for i in range(torch.cuda.device_count()):
           print(torch.cuda.get_device_properties(i).name)
    torch.cuda.set_device(1)'''
    quantization_config = BitsAndBytesConfig(
        load_in_8bit=True
    )
    model = LlavaOnevisionForConditionalGeneration.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf", torch_dtype=torch.float16, device_map="auto")
    model.eval()
    #print(model.device)
    processor = AutoProcessor.from_pretrained("llava-hf/llava-onevision-qwen2-7b-ov-hf")
    #setup our one-shot example
    eg_img = Image.open(f"FewShotEgs/{environment_name}.jpg") 
    eg_img = np.array(eg_img)
    eg_txt = [utils.get_example(environment_name)]
    if environment_name == "snake-v0":
        env = gym.make('snake-v0')
        state = env.reset()
    else:
        env = gym.make(environment_name, render_mode='rgb_array')
        state, info = env.reset()
    done = False
    '''for i in range(35):
        action = np.random.randint(low = 0, high = 4)
        state, _, _, _, _ = env.step(action)'''
    i = 0
    while not done:
        if environment_name == 'snake-v0':
            image = state
        else:
            image = env.render()
        im = Image.fromarray(image)
        im.save(f"agent_outputs/{environment_name}_{i}.jpg")
        
        images = [image]#[eg_img, image]
        chat = utils.make_conversation(0, environment_name,"") #eg_txt)
        prompt = processor.apply_chat_template(chat, add_generation_prompt=True)
        inputs = processor(images=images, text=prompt, return_tensors="pt").to("cuda:0", torch.float16)
        output = model.generate(**inputs, max_new_tokens=5)
        instruction = processor.decode(output[0], skip_special_tokens=True)
        action = utils.decode(environment_name, instruction[len(inputs) : ].lower())
        print(action)
        print(instruction)
        if action == -1:
            print("bollocks")
            action = np.random.randint(low = 0, high = 4)
        state, reward, done, trunc = env.step(action)
        print(state)
        print(reward)
        print(done)
        print(trunc)
        i += 1
    
    return
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--env',type=str, default = 'snake-v0', help='Which environment to use. Currently supports snake and lunar lander.')
    args = parser.parse_args()
    if args.env not in ["LunarLander-v2", "snake-v0"]:
        print("Environment not supported!!!")
    else:
        run_hf(args.env)