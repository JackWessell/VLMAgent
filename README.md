# VLMAgent

This repository demonstrates the ability of large multimodal models like claude to stand in for policies in general reinforcment learning environments.
To begin, first run 
```shell
git clone https://github.com/JackWessell/VLMAgent.git
conda env create --name my-env-name --file environment.yml
```
Then, generate an API account at the following: https://docs.anthropic.com/en/api/getting-started
Finally, run: 
```shell
python vlm.py --env ENVIRONMENT --key ANTHROPIC_API_KEY
```
Currently, only two environments are supported: snake-v0, and LunarLander-v2.