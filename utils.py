#can add different prompts for different tasks. For our project, we can describe different environments and their respective goals.
topics = []
def make_prompt(text, topic):
    prompt = topics[topic].format(desc = text)
    user =   {
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": prompt },
            ],
    }
    return user
    
#generate a conversation for our in-context few-shot learning. This process can be made more complex in the future,
#but for now a conversation will be built from a base prompt and a collection of description-answer pairs.
def make_conversation(n_shots, answers):
    #configure a base prompt here. We can handle few-shot learning with this function if needed.
    def begin_conversation():
        system = {
        "role": "system",
        "content": [
            {"type": "text", "text": """A chat between a human and a helpful AI assistant. 
                                    The human needs help landing the the space probe safely and the assistant provides the best action at each step.
                                    The goal is to land the probe safely on the surface of the moon and, preferably, between the flags.
                                    At each step, the assistant can instruct the human to fire the main thruster, fire the left thruster, 4
                                    fire the right thruster, or do nothing. The human cannot fire multiple thrusters at once."""},
            ],
        }
        return system 
    base_prompt = "The image depicts the current state of my lunar lander. What action should I take to most improve my chances of landing safely between the flags? Responsd with either \'fire left\', \'fire right\', \'fire main\', \'or do nothing.\'"
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