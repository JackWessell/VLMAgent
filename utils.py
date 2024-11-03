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
def make_conversation(descriptions, answers):
    #configure a base prompt here. We can handle few-shot learning with this function if needed.
    base_prompt = ""
    def make_prompt(text):
        prompt = base_prompt.format(desc = text)
        user =   {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": prompt },
                ],
        }
        return user
    def make_answer(ans):
        answer = {
            "role": "assistant",
            "content": [
                    {"type": "text", "text": ans},
                    ],
            }
        return answer
    conversation = []
    for i in range(len(descriptions)-1):
        conversation.append(make_prompt(descriptions[i]))
        conversation.append(make_answer(answers[i]))
    conversation.append(make_prompt(descriptions[-1]))
    return conversation