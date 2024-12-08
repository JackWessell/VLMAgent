#can add different prompts for different tasks. For our project, we can describe different environments and their respective goals.
system_prompts = {
    "Test" : "A chat between a human and a helpful AI assistant.",
    "LunarLander-v2" : """You are a helpful AI assistant that is an expert in playing games. The user is playing a game in which it must safely land a lunar probe by firing 
    different rockets. Help the user make the best decision given the current position of the lander relative to the goal and how it is moving as described by the prompt.""",

    "snake-v0" : """You are a helpful AI assistant that is an expert in playing games. The user you are working with needs help playing the popular computer game snake.
                    In snake, the goal is to eat as much food as possible. Each time the snake eats a piece of food, its length increases. Help the human make the best 
                    move to eat more food without the snake going out of bounds or running into its own tail."""
    }
base_prompts = {
    "Test" : "Can you describe to me what you see in the image? What do you think is happening?",
    
     "LunarLander-v2" : """The image provided depicts the current position of my lunar lander relative to the goal, which is the area between the two flags. The flags are 
     at coordinate 0 (between -1.5 and 1.5) and my ship is at the horizontal position {}.
     Currently, my lander is {} at a speed of {:.2f}, {} at a speed of {:.2f}, and is {} at a rotational speed of {:.2f} with a current angle of {:.2f} relative to the horizontal.
       I need to make sure that the ship is not descending too fast while remaining relatively stable both horizontally and angularly. Try to avoid ascending if possible as this moves us further from out goal.
        Given the information I provided as well as what you can determine from the image, what should I do now to ensure a safe landing between the two flags?
        I can either fire my left booster, fire my right booster, fire my main booster, or do nothing. I can only fire one booster at a time, and how it effects the ship's
        trajectory is determined by the ship's orientation as it descends. To combat clockwise rotation fire the right booster, while to combat counterclockwise rotation fire the left booster.
        Please include your answer in single quotes so I can extract it easily like so: \'fire main\'.""",
   
    "snake-v0" : """The image provided depicts the current state of my game of snake. The head of my snake is red, the tail is black, and food is blue. The board is a {} by {} grid, with (0, 0) representing the top left corner and ({}, {}) being
    the bottom right. Increasing the first coordinate corresponds to moving down, while increasing the second coordinate corresponds to moving right. 
    At the moment, my snake's head is at ({}, {}) and it is moving in the direction: {}. The food is {} units {} and {} units {} the current position of the snake's head.
    In what direction should I move my snake in order to eat the food while avoiding both my tail and going out of bounds?
    Include your answer in single quotes so that I can easily extract it like so: \'down\'.
                """
    }
def get_example(environment):
    if environment == "LunarLander-v2":
        return "Fire Right."
    if environment == "snake-v0":
        return "left"

def encode_snake(instruction):
    if instruction == 0:
        return "up"
    if instruction == 1:
        return "right"
    if instruction == 2:
        return "down"
    if instruction == 3:
        return "left"
    else:
        return -1
def decode(environment, instruction):
    if environment == "LunarLander-v2":
        if "do nothing" in instruction or "none" in instruction or "nothing" in instruction:
            return 0
        if "fire left" in instruction:
            return 3
        if "fire main" in instruction:
            return 2
        if "fire right" in instruction:
            return 1
        else:
            return -1
    if environment == "snake-v0":
        if "up" in instruction:
            return 0
        if "right" in instruction:
            return 1
        if "down" in instruction:
            return 2
        if "left" in instruction:
            return 3
        else:
            return -1
#generate a conversation for our in-context few-shot learning. This process can be made more complex in the future,
#but for now a conversation will be built from a base prompt and a collection of description-answer pairs.
#info is a tuple of environment information that we can inject into the base prompt to help out the model.
def find_dist(x1, y1, x2, y2):
    x = x2 - x1
    y = y2 - y1
    if x < 0 and y < 0:
        return (-x, "to the left", -y, "above")
    if x < 0 and y >= 0:
        return (-x, "to the left", y, "below")
    if x >= 0 and y < 0:
        return (x, "to the right", -y, "above")
    else:
        return(x, "to the right", y, "below")
#extract state information to embed into the text prompt. Re-factoring - include for each environment so that we can appropriately inject state information into the prompts
def get_state(environment, state):
    if environment == "LunarLander-v2":
        hor_vel = state[2]
        ver_vel = state[3] 
        angle = state[4]
        ang_vel = state[5]
        if hor_vel < -.1:
            hor_vel = -1*hor_vel
            hor_dir = "drifting left"
        elif hor_vel > .1:
            hor_dir = "drifting right"
        else:
            hor_dir = "not drifting much horizontally"
            
        if ver_vel < -.5:
            ver_dir = "falling very fast"
        elif ver_vel < -.25:
            ver_dir = "falling quickly"
        elif ver_vel < 0:
            ver_dir = "descending stably"
        else:
            ver_dir = "moving upwards"
            
        if ang_vel < 0:
            ang_dir = "rotating clockwise"
        elif ang_vel > 0:
            ang_dir = "rotating counterclockwise"
        else:
            ang_dir = "completely flat"
        #added the x coordinate to help Claude determine how close the lander is to the goal
        return(hor_dir, hor_vel, ver_dir, abs(ver_vel), angle, ang_vel, ang_dir, state[0])
def make_conversation(n_shots, environment, answers, info, media, images):
    #configure a base prompt here. We can handle few-shot learning with this function if needed.
    #base_prompt = base_prompts[environment]
    system_prompt = system_prompts[environment]
    def begin_conversation():
        system = {
        "role": "system",
        "content": [
            {"type": "text", "text": system_prompt},
            ],
        }
        return system 

    def make_prompt(media_type, data, text):
        if environment == "snake-v0":
            snake_x = text[1][0]
            snake_y = text[1][1]
            food_x = text[3][0]
            food_y = text[3][1]
            res = find_dist(snake_x, snake_y, food_x, food_y)
            prompt = base_prompts[environment].format(text[0][1], text[0][0], text[0][1]-1, text[0][0]-1, text[1][1], text[1][0], text[2], res[0], res[1], res[2], res[3])
        elif environment == "LunarLander-v2":
            prompt = base_prompts[environment].format(text[7], text[0], text[1], text[2], text[3], text[6], text[5], text[4])
        user =   {
            "role": "user",
            "content": [
                {"type": "image",
                 "source": {
                        "type": "base64",
                        "media_type": media_type,
                        "data": data,
                    },
                 },
                {"type": "text", "text": prompt },
                ],
        }
        return user
    #make_answer only needed in a few-shot setting
    def make_answer(ans):
        answer = {
            "role": "assistant",
            "content": [
                    {"type": "text", "text": ans},
                    ],
            }
        return answer
    
    conversation = []
    conversation.append(begin_conversation())
    #we can add additional information to our prompt and few shot examples.
    for i in range(n_shots):
        conversation.append(make_prompt(info[i], media[i], images[i]))
        conversation.append(make_answer(answers[i]))
    conversation.append(make_prompt(media[-1], images[-1], info[-1]))
    return conversation