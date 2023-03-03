from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain


def get_model():
    model = OpenAI(temperature = 0, model_name= "text-davinci-003")
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
