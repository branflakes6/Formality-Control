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


def gold_labels(data, n_samples):
    data = data[:n_samples]
    english = []
    kormal = []
    informal = []
    for x in data.iterrows():
        english.append(x[1]['English'])
        kormal.append(x[1]['Formal_Korean'])

    f = open("formal_gold", "w", encoding='utf-8')
    for x in kormal:
        f.write(x)

    f = open("eng_gold", "w", encoding='utf-8')
    for x in english:
        f.write(x)

    f = open("informal_gold", "w", encoding='utf-8')
    for x in informal:
        f.write(x)


def english_to_formal_prompt_n_shot(data, n, query):
    data = data[:n]
    formal_kr = data['Formal_Korean']
    english = data['English']
    prompt = ''
    for i, _ in enumerate(formal_kr):
        if english[i] != query:
            prompt = prompt + "English:" + english[i] + 'Formal Korean:' + formal_kr[i]
    prompt = prompt + '{query}'
    return prompt


def formal_to_informal_prompt_n_shot(data, n, query):
    data = data[:n]
    formal_kr = data['Formal_Korean']
    informal_kr = data['Informal_Korean']
    prompt = ''
    for i, _ in enumerate(formal_kr):
        if formal_kr[i] != query:
            prompt = prompt + "Formal Korean:" + formal_kr[i] + 'Informal Korean:' + informal_kr[i]
    prompt = prompt + '{query}'
    return prompt


def prompter(prompt, query):
    prompt = PromptTemplate(
        input_variables=['query'],
        template=prompt
    )
    chain = LLMChain(llm=LLM, prompt=prompt)
    prediction = chain.run(query)
    return prediction


def eng_to_formal_n_shot(data, n_samples, n_shot):
    data = data[:n_samples]
    preds = []
    for x in tqdm(data.iterrows()):
        query = x[1]['English']
        query = 'English:' + query + 'Formal Korean:'
        if n_shot > 0:
            prompt = english_to_formal_prompt_n_shot(data, n_shot, x)
        else:
            prompt = '{query}'
        pred = prompter(prompt, query)
        pred = pred + "\n"
        preds.append(pred)

    f = open("preds_formal)", "w", encoding='utf-8')
    for x in preds:
        f.write(x)
    data = pd.read_csv('data/train/en-ko/en-kr_combined_annotated')
    gold_labels(data, n_samples)


class FormalToInformal(object):
    def __init__(self, data, formal_lang, informal_lang, prompt_func, file_path):
        self.data = data
        self.formal_lang = formal_lang
        self.informal_lang = informal_lang
        self.prompt_func = prompt_func
        self.file_path = file_path

    def zero_shot(self, zero_prompt, n_samples):
        data = self.data[:n_samples]
        preds = []
        for x in tqdm(data.iterrows()):
            query = x[1][self.formal_lang]
            print("prompt", zero_prompt)
            print("query", query)
            # pred = prompter(zero_prompt, query)
            # print("pred", pred)
            # print("DONE")
            # pred = pred + "\n"
            # preds.append(pred)

    def n_shot(self, n_samples):
        data = self.data[:n_samples]
        preds = []
        for x in tqdm(data.iterrows()):
            query = x[1][self.formal_lang]
            prompt, query = self.prompt_func(data, x, query)
            print("prompt", prompt)
            print("query", query)
            print("DONE")
            pred = prompter(prompt, query)
            print("pred", pred)
            pred = pred + "\n"
            preds.append(pred)

        # f = open("preds_informal", "w", encoding='utf-8')
        # for x in preds:
        #     f.write(x)
        # data = pd.read_csv('data/train/en-ko/en-kr_combined_annotated')
        # gold_labels(data, n_samples)


def zero_shot_korean_formal_to_informal():
    data = pd.read_csv('data/train/en-ko/en-kr_combined')

    def prompt_1(data, x, query):
        return '', ''

    korean_zero = FormalToInformal(data, 'Formal_Korean', 'Informal_Korean', prompt_1, '')
    prompt = 'Can you convert this from Formal Korean to Informal Korean: {query}'
    korean_zero.zero_shot(prompt, 3)
    prompt = 'Can you give me the informal Korean of the following formal Korean sentence in Korean Hangul: {query}'
    korean_zero.zero_shot(prompt, 1)
    prompt = 'Can you give me the informal Korean of the following formal Korean sentence in Korean, please provide the output in the Korean language: {query}'
    korean_zero.zero_shot(prompt, 1)


def n_shot_korean_formal_to_informal():
    data = pd.read_csv('data/train/en-ko/en-kr_combined')

    def prompt_1(data, x, query):
        prompt = formal_to_informal_prompt_n_shot(data, 5, x)
        query = 'Formal Korean:' + query + 'Informal Korean:'
        return prompt, query

    kr_form_to_inform = FormalToInformal(data, 'Formal_Korean', 'Informal_Korean', prompt_1, '')
    kr_form_to_inform.n_shot(2)

    def prompt_2(data, x, query):
        prompt = formal_to_informal_prompt_n_shot(data, 5, x)
        query = 'Korean:' + query + 'Korean:'
        return prompt, query

    kr_form_to_inform.prompt_func = prompt_2
    kr_form_to_inform.n_shot(2)

    def prompt_3(data, x, query):
        prompt = formal_to_informal_prompt_n_shot(data, 5, x)
        query = 'Korean:' + query + 'Korean:'
        return prompt, query

    kr_form_to_inform.prompt_func = prompt_3
    kr_form_to_inform.n_shot(2)

