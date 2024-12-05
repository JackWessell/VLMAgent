#can add different prompts for different tasks. For our project, we can describe different environments and their respective goals.
system_prompts = {
    "Test" : "A chat between a human and a helpful AI assistant.",
    "LunarLander-v2" : """A chat between a human and a helpful AI assistant. 
                            The human needs help landing the the space probe safely and the assistant provides the best action at each step.
                            The goal is to land the probe safely on the surface of the moon and, preferably, between the flags.
                            At each step, the assistant can instruct the human to fire the main thruster, fire the left thruster, 4
                            fire the right thruster, or do nothing. The human cannot fire multiple thrusters at once.""",
    "snake-v0" : """ A chat between a human and a helpful AI assistant. The human is trying to play the popular game snake and needs help understanding
                    what move to make at each point."""
    }
base_prompts = {
    "Test" : "Can you describe to me what you see in the image? What do you think is happening?",
     "LunarLander-v2" : """The image depicts the current position of my lunar lander. What action should I take to most improve my chances of landing safely between the flags? Respond with                            either \'fire left\', \'fire right\', \'fire main\', \'or do nothing.\'""",
    "snake-v0" : """The image depicts the current state of my game of snake. In what direction should the snake, with its red head and black body, move to eat the blue cube? Please explain your answer given what you know about the game of snake. Include details such as where the snake is relative to the cube in your answer."""
    }
def get_example(environment):
    if environment == "LunarLander-v2":
        return "Fire Right."
    if environment == "snake-v0":
        return "left"

def decode(environment, instruction):
    if environment == "LunarLander-v2":
        if "do nothing" in instruction:
            return 0
        if "fire left" in instruction:
            return 1
        if "fire main" in instruction:
            return 2
        if "fire right" in instruction:
            return 3
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
def make_conversation(n_shots, environment, answers):
    #configure a base prompt here. We can handle few-shot learning with this function if needed.
    base_prompt = base_prompts[environment]
    system_prompt = system_prompts[environment]
    def begin_conversation():
        system = {
        "role": "system",
        "content": [
            {"type": "text", "text": system_prompt},
            ],
        }
        return system 

    def make_prompt(text = None):
        if text is not None:
            prompt = base_prompt.format(desc = text)
        else:
            prompt = base_prompt
        user =   {
            "role": "user",
            "content": [
                {"type": "image"},
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
        conversation.append(make_prompt())
        conversation.append(make_answer(answers[i]))
    conversation.append(make_prompt())
    return conversation