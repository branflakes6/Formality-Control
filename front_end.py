import streamlit as st
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LLM = OpenAI(temperature = 0, model_name= "text-davinci-003")
st.title('Korean Formality Control')


def run(query):
    with st.spinner('Translating'):
        prompt = PromptTemplate(
                input_variables=["query"],
                template="English: Yeah Did your mom know you were throwing the party?\n Formal Korean: 그, 어머님은 [F]님이[/F] 그 파티 연 거 [F]아세요[/F]?\n Informal Korean: 그, 어머님은 [F]네가[/F] 그 파티 연 거 [F]아셔[/F]?\n English: [F]내가[/F] [F]너[/F] 잘못되라고 하고 있다는 [F]거야[/F]?\n Formal Korean:[F]제가[/F] [F]님[/F] 잘못되라고 하고 있다는 [F]거예요[/F]?\n Informal Korean:[F]내가[/F] [F]너[/F] 잘못되라고 하고 있다는 [F]거야[/F]?\n English: {query} "
            )
        chain = LLMChain(llm=LLM, prompt=prompt)
        formal = chain.run(query)

    with st.spinner('Converting'):
        prompt = PromptTemplate(
            input_variables=["formal"],
            template="Formal Korean: 그, 어머님은 [F]님이[/F] 그 파티 연 거 [F]아세요[/F]?\n Informal Korean: 그, 어머님은 [F]네가[/F] 그 파티 연 거 [F]아셔[/F]?\n Formal Korean:[F]제가[/F] [F]님[/F] 잘못되라고 하고 있다는 [F]거예요[/F]?\n Informal Korean:[F]내가[/F] [F]너[/F] 잘못되라고 하고 있다는 [F]거야[/F]?\n Formal Korean: {sentence} "
        )
        chain = LLMChain(llm=LLM, prompt=prompt)
        informal = chain.run(formal)
    return formal, informal



input = st.text_input('Enter an English sentence to translate', '')
if st.button('Translate'):
    formal, informal = run(input)
    st.write("Formal Korean", formal)
    st.write("Informal Korean", informal)
