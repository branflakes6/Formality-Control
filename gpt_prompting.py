from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LLM = OpenAI(temperature=0, model_name="text-curie-001")


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


def english_to_formal_prompt(data, n, variable):
    data = data[:n]
    formal_kr = data['Formal_Korean']
    english = data['English']
    prompt = ' '
    for i, _ in enumerate(formal_kr):
        if english[i] != query:
            prompt = prompt + "English:" + english[i] + 'Formal Korean:' + formal_kr[i]
    prompt = prompt + '{query}'
    return prompt


# Prompt 'Formal Korean: formal, Informal Korean: informal'
def formal_to_informal_prompt(data, n, query):
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
    return prediction

def formal_to_informal(data):
    data = data[:5]
    preds = []
    for x in tqdm(data.iterrows()):
        query = x[1]['Formal_Korean']
        prompt = formal_to_informal_prompt(data, 5, x)
        query = 'Formal Korean:' + query + 'Informal Korean:'
        print(query)
        pred = prompt_test(prompt, query)
        pred = pred + "\n"
        preds.append(pred)

    f = open("preds_informal", "w", encoding='utf-8')
    for x in preds:
        f.write(x)

    data = pd.read_csv('data/train/en-ko/en-kr_combined_annotated')
    data = data[:1]
    inf = []
    kor = []
    for x in data.iterrows():
        inf.append(x[1]['Informal_Korean'])
        kor.append(x[1]['Formal_Korean'])

    f = open("kor", "w", encoding='utf-8')
    for x in kor:
        f.write(x)

    f = open("inf", "w", encoding='utf-8')
    for x in inf:
        f.write(x)


def eng_to_formal(data):
    preds = []
    for x in tqdm(data.iterrows()):
        query = x[1]['English']
        prompt = english_to_formal_prompt(data, 5, x)
        query = 'English:' + query + 'Formal Korean:'
        print(query)
        print(prompt)
        pred = prompt_test(prompt, query)
        pred = pred + "\n"
        preds.append(pred)

    f = open("preds_formal)", "w", encoding='utf-8')
    for x in preds:
        f.write(x)

    data = pd.read_csv('data/train/en-ko/en-kr_combined_annotated')
    data = data[:1]
    eng = []
    kor = []
    for x in data.iterrows():
        eng.append(x[1]['English'])
        kor.append(x[1]['Formal_Korean'])

    f = open("formal_kr", "w", encoding='utf-8')
    for x in kor:
        f.write(x)

    f = open("eng_gold", "w", encoding='utf-8')
    for x in eng:
        f.write(x)


data = pd.read_csv('data/train/en-ko/en-kr_combined')
eng_to_formal(data)
formal_to_informal(data)

