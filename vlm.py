#NOTE: The code below is configured to make running the code possible in Pace-ICE ondemand. You should either change the lines below to reflect your virtual env or comment them out.
import sys
'''
sys.executable = 'miniconda3/envs/DRL/bin/python3.10'
sys.path = ['/home/hice1/jwessell6/DRL/VLMAgent/Gym-Snake', '/home/hice1/jwessell6/miniconda3/envs/DRL/lib/python3.10', 
    '/home/hice1/jwessell6/miniconda3/envs/DRL/lib/python3.10/site-packages', '/home/hice1/jwessell6/miniconda3/envs/DRL/lib/python3.10/lib-dynload', '/home/hice1/jwessell6/.local/bin']
'''
sys.path += ['Gym_Snake']
import base64
import numpy as np
import torch 
#from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration, BitsAndBytesConfig
import pickle
import argparse
import gym
import utils 
from PIL import Image
import anthropic
import os
import gym_snake
import re
import time

def run_hf(environment_name, api_key):
    os.environ['ANTHROPIC_API_KEY'] = api_key
    client = anthropic.Anthropic()
    #setup our one-shot example
    eg_img = Image.open(f"FewShotEgs/{environment_name}.jpg") 
    eg_img = np.array(eg_img)
    eg_txt = [utils.get_example(environment_name)]
    
    if environment_name == "snake-v0":
        grid_size = [10,10]
        env = gym.make('snake-v0', grid_size = grid_size, unit_size = 35)
        state = env.reset()
    else:
        env = gym.make(environment_name, render_mode='rgb_array')
        state, info = env.reset()
        for i in range(10):
            action = np.random.randint(low = 0, high = 4)
            state, _, _, _, _ = env.step(action)
        
    done = False
    idx = 0
    res = []
    while not done:
        if environment_name == 'snake-v0':
            image = state
        else:
            image = env.render()
        im = Image.fromarray(image)
        im.save(f"agent_outputs/{environment_name}/{idx}.jpg")
        with open(f"agent_outputs/{environment_name}/{idx}.jpg", "rb") as image_file:
            enc = base64.b64encode(image_file.read()).decode('utf-8')
        
        if environment_name == "LunarLander-v2":   
            curr_info = utils.get_state(environment_name, state)
        else:
            for i in range(grid_size[0]):
                for j in range(grid_size[1]):
                    if env.controller.grid.food_space((i,j)):
                        food = [i, j]
                        break
            curr_info = (grid_size, env.controller.snakes[0].head, utils.encode_snake(env.controller.snakes[0].direction), food)
        chat = utils.make_conversation(0, environment_name, [], [curr_info], ["image/jpeg"], [enc])
        try:
            message = client.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=1000,
                temperature=0,
                system=chat[0]['content'][0]['text'],
                messages=[
                    chat[1]
                ]
            )
        except:
            print("Backing off...")
            time.sleep(2)
            continue
        pattern = r"'[^']{2,10}'"
        string = message.content[0].text
        matches = re.findall(pattern, string)
        action = utils.decode(environment_name, matches[0][1:-1])
        if action == -1:
            #try again
            print(matches)
            print(string)
            continue
        if environment_name == "LunarLander-v2":
            state, reward, done, trunc, _ = env.step(action)
        else:
            state, reward, done, trunc = env.step(action)
        res.append((chat[1], message.content[0].text, reward))
        print(idx)
        idx += 1
        with open(f"agent_outputs/{environment_name}/results.pkl", "wb") as file:
            pickle.dump(res,file)
    return
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument('--env',type=str, default = 'snake-v0', help='Which environment to use. Currently supports snake and lunar lander.')
    parser.add_argument('--key', type=str, help='Anthropic API key')
    args = parser.parse_args()
    if args.env not in ["LunarLander-v2", "snake-v0"]:
        print("Environment not supported!!!")
    else:
        run_hf(args.env, args.key)