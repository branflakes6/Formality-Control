from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
import pandas as pd

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LLM = OpenAI(temperature=0, model_name="text-davinci-003")


def get_model():
    model = OpenAI(temperature=0, model_name="text-davinci-003")
    return model


def get_prompt(data, n, variable):
    return data[0]


def training_loop(data, option):
    sentence = option
    prompt = get_prompt(data, 1, sentence)
    prompt = PromptTemplate(
        input_variables=["sentence"],
        template=prompt
    )
    chain = LLMChain(llm=LLM, prompt=prompt)
    prediction = chain.run(sentence)
    prediction = prediction[18:]
    print("prediction", prediction)
    return prediction


def hindi_translater(sent):
    prompt = "Translate from Hindi to English: {sent}"
    prompt = PromptTemplate(
        input_variables=["sent"],
        template=prompt
    )
    chain = LLMChain(llm=LLM, prompt=prompt)
    out = chain.run(sent)
    return out


# Loop through n rows of the data
# Create a string in the form 'English: english, Formal Korean: formal, Informal Korean: inforaml'
# Add all strings together
# Ensure variable is not in the prompt (Cant have the sentence we are translating in our prompt or model will know the answer)
def english_to_formal_informal_prompt(data, n, variable):
    return 0


# Prompt 'Formal Korean: formal, Informal Korean: informal'
def formal_to_formal_prompt(data, n, query):
    data = data[:n]
    formal_kr = data['Formal_Korean']
    informal_kr = data['Informal_Korean']
    prompt = ''
    for i, _ in enumerate(formal_kr):
        if formal_kr[i] != query:
            prompt = prompt + "Formal Korean:" + formal_kr[i] + 'Informal Korean:' + informal_kr[i]
    prompt = prompt + '{query}'
    return prompt


def prompt_test(prompt, query):
    prompt = PromptTemplate(
        input_variables=['query'],
        template=prompt
    )
    chain = LLMChain(llm=LLM, prompt=prompt)
    prediction = chain.run(query)
    print("prediction", prediction)
    return prediction


data = pd.read_csv('data/train/en-ko/en-kr_combined_annotated')
query = data['Formal_Korean'][0]
prompt = formal_to_formal_prompt(data, 10, query)
query = 'Formal Korean:' + query + 'Informal Korean:'

pred = prompt_test(prompt, query)
