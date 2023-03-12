from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
import pandas as pd
from tqdm import tqdm
import datasets

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LLM = OpenAI(temperature=0, model_name="text-curie-001")


def gold_labels_kr(data, n_samples):
    data = data[:n_samples]
    english = []
    formal = []
    informal = []
    for x in data.iterrows():
        english.append(x[1]['English'])
        formal.append(x[1]['Formal_Korean'])
        informal.append(x[1]['Informal_Korean'])

    f = open("outputs\\formal_gold", "w", encoding='utf-8')
    for x in formal:
        f.write(x.strip()+'\n', )

    f = open("outputs\eng_gold", "w", encoding='utf-8')
    for x in english:
        f.write(x.strip()+'\n')

    f = open("outputs\informal_gold", "w", encoding='utf-8')
    for x in informal:
        f.write(x.strip()+'\n')

def gold_labels_vi(data, n_samples):
    data = data[:n_samples]
    english = []
    formal = []
    informal = []
    for x in data.iterrows():
        english.append(x[1]['English'])
        formal.append(x[1]['Formal_Viet'])
        informal.append(x[1]['Informal_Viet'])

    f = open("outputs\\formal_gold_vi", "w", encoding='utf-8')
    for x in formal:
        f.write(x.strip()+'\n', )

    f = open("outputs\eng_gold_vi", "w", encoding='utf-8')
    for x in english:
        f.write(x.strip()+'\n')

    f = open("outputs\informal_gold_vi", "w", encoding='utf-8')
    for x in informal:
        f.write(x.strip()+'\n')

def prompter(prompt, query):
    prompt = PromptTemplate(
        input_variables=['query'],
        template=prompt
    )
    chain = LLMChain(llm=LLM, prompt=prompt)
    prediction = chain.run(query)
    return prediction


class GPTPrompter(object):
    def __init__(self, data, formal_lang, prompt_func, file_path, n):
        self.data = data
        self.formal_lang = formal_lang
        self.prompt_func = prompt_func
        self.file_path = file_path
        self.n = n
    def zero_shot(self, zero_prompt, n_samples):
        data = self.data[:n_samples]
        preds = []
        for x in tqdm(data.iterrows()):
            query = x[1][self.formal_lang]
            pred = prompter(zero_prompt, query)
            preds.append(pred)


        f = open(self.file_path, "w", encoding='utf-8')
        for x in preds:
            f.write(x)

    def n_shot(self, n_samples):
        data = self.data[:n_samples]
        preds = []
        for x in tqdm(data.iterrows()):
            query = x[1][self.formal_lang]
            prompt, query = self.prompt_func(data, self.n, query)
            print("prompt", prompt)
            print("query", query)
            pred = prompter(prompt, query)
            pred = pred.strip()
            pred = pred + '\n'
            preds.append(pred)

        f = open(self.file_path, "w", encoding='utf-8')
        for x in preds:
            f.write(x)


def english_to_formal_prompt_n_shot(data, n, query):
    data = data.sample(frac=1).reset_index()
    data = data[:n]
    formal_kr = data['Formal_Korean']
    english = data['English']
    prompt = ''
    for i, _ in enumerate(formal_kr):
        if english[i] != query:
            prompt = prompt + "Translate English: \"" + english[i] + '\" To Korean: ' + formal_kr[i]
    prompt =  prompt + '{query}'
    return prompt


def formal_to_informal_prompt_n_shot(data, n, query):
    data = data.sample(frac=1).reset_index()
    data = data[:n]
    formal_kr = data['Formal_Korean']
    informal_kr = data['Informal_Korean']
    prompt = ''
    for i, _ in enumerate(formal_kr):
        if formal_kr[i] != query:
            prompt = prompt + "Formal Korean:" + formal_kr[i] + 'Informal Korean:' + informal_kr[i]
    prompt = prompt + '{query}'
    return prompt


def zero_shot_eng_to_korean(n_samples):
    data = pd.read_csv('data/train/en-ko/en-kr_combined')

    def prompt_1(data, x, query):
        return '', ''

    korean_zero = GPTPrompter(data, 'English', prompt_1, '.\outputs\eng_kr_zero', 0)
    prompt = 'Can you translate the the following English into Korean: {query}'
    korean_zero.zero_shot(prompt, n_samples)


def eng_to_formal_kr(n_samples, n_shot):
    data = pd.read_csv('data/train/en-ko/en-kr_combined')

    def prompt_1(data, n, query):
        prompt = english_to_formal_prompt_n_shot(data, n, query)
        query = 'Translate English : \"' + query + '\" To Korean :'
        return prompt, query

    eng_to_formal = GPTPrompter(data, 'English', prompt_1, '.\outputs\eng_formal', n_shot)
    eng_to_formal.n_shot(n_samples)

    data = pd.read_csv('data/train/en-ko/en-kr_combined_annotated')
    gold_labels_kr(data, n_samples)


def zero_shot_korean_formal_to_informal(n_samples):
    data = pd.read_csv('data/train/en-ko/en-kr_combined')

    def prompt_1(data, x, query):
        return '', ''

    korean_zero = GPTPrompter(data, 'Formal_Korean', prompt_1, '.\outputs\kr_informal_zero', 0)
    prompt = 'Can you convert this from Formal Korean to Informal Korean do not included newlines: {query}'
    korean_zero.zero_shot(prompt, n_samples)

    korean_zero.file_path = 'zero_preds2'
    prompt = 'Can you give me the informal Korean of the following formal Korean sentence in Korean Hangul: {query}'
    korean_zero.zero_shot(prompt, n_samples)

    korean_zero.file_path = 'zero_preds3'
    prompt = 'Can you translate the following Formal Korean sentence into Informal Korean, please provide the output in Korean: {query}'
    korean_zero.zero_shot(prompt, n_samples)

    data = pd.read_csv('data/train/en-ko/en-kr_combined_annotated')
    gold_labels_kr(data, n_samples)


def n_shot_korean_formal_to_informal(n_samples, n_shot):
    data = pd.read_csv('data/train/en-ko/en-kr_combined')

    def prompt_1(data, n, query):
        prompt = formal_to_informal_prompt_n_shot(data, n, query)
        query = 'Formal Korean:' + query + 'Informal Korean:'
        return prompt, query

    kr_form_to_inform = GPTPrompter(data, 'Formal_Korean', prompt_1, '.\outputs\kr_informal', n_shot)
    kr_form_to_inform.n_shot(n_samples)

    data = pd.read_csv('data/train/en-ko/en-kr_combined_annotated')
    gold_labels_kr(data, n_samples)


def calc_bleu():
    metric = datasets.load_metric('sacrebleu')
    data = pd.read_csv('data/train/en-ko/en-kr_combined')

    f = open('outputs/eng_formal', "r", encoding='utf-8')
    data = f.readlines()
    preds = []
    for x in data:
        preds.append([x])

    data = pd.read_csv('data/train/en-ko/en-kr_combined')
    data = data['Formal_Korean'][:len(preds)]
    gold_labels = []
    for x in data:
        gold_labels.append([x])

    print(gold_labels)
    result = metric.compute(predictions=preds, references=gold_labels)
    print(result)


#eng_to_formal_kr(20, 5)
#zero_shot_eng_to_korean(30)
#zero_shot_korean_formal_to_informal(10)
#n_shot_korean_formal_to_informal(399, 5)
#data = pd.read_csv('data/train/en-ko/en-kr_combined_annotated')
#gold_labels_kr(data, 10)
#calc_bleu()

def viet_to_formal_prompt_n_shot(data, n, query):
    data = data.sample(frac=1).reset_index()
    data = data[:n]
    formal_vi = data['Formal_Viet']
    english = data['English']
    prompt = ''
    for i, _ in enumerate(formal_vi):
        if english[i] != query:
            prompt = prompt + "Translate English: \"" + english[i] + '\" To Vietnamese: ' + formal_vi[i]
    prompt =  prompt + '{query}'
    return prompt


def eng_to_formal_vi(n_samples, n_shot):
    data = pd.read_csv('data/train/en-vi/en-vi_combined')

    def prompt_1(data, n, query):
        prompt = viet_to_formal_prompt_n_shot(data, n, query)
        query = 'Translate English : \"' + query + '\" To Vietnamese :'
        return prompt, query

    eng_to_formal = GPTPrompter(data, 'English', prompt_1, '.\outputs\eng_formal_vi', n_shot)
    eng_to_formal.n_shot(n_samples)

    data = pd.read_csv('data/train/en-vi/en-vi_combined-annotated')
    gold_labels_vi(data, n_samples)


def calc_bleu_vi():
    metric = datasets.load_metric('sacrebleu')

    f = open('outputs/eng_formal_vi', "r", encoding='utf-8')
    data = f.readlines()
    preds = []
    for x in data:
        preds.append([x])

    data = pd.read_csv('data/train/en-vi/en-vi_combined')
    data = data['Formal_Viet'][:len(preds)]
    gold_labels = []
    for x in data:
        gold_labels.append([x])

    print(gold_labels)
    result = metric.compute(predictions=preds, references=gold_labels)
    print(result)

eng_to_formal_vi(20, 5)
calc_bleu_vi()