import os
from dotenv import load_dotenv
import streamlit as st
import gpt_prompting

load_dotenv()
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

data = [
    'Formal Korean: 그, 어머님은 [F]님이[/F] 그 파티 연 거 [F]아세요[/F]?\n Informal Korean: 그, 어머님은 [F]네가[/F] 그 파티 연 거 [F]아셔[/F]?\n Formal Korean:[F]제가[/F] [F]님[/F] 잘못되라고 하고 있다는 [F]거예요[/F]?\n Informal Korean:[F]내가[/F] [F]너[/F] 잘못되라고 하고 있다는 [F]거야[/F]?\n Formal Korean: {sentence}']

st.title('Formality Control')

option = st.selectbox('Pick a sample sentence to translate', ('그게 [F]님이[/F] 제일 [F]좋아하시는[/F] [F]피자예요[/F]?', '무슨 이유에서든 컴퓨터에 공간이 충분하지 않다는 [F]뜻이에요[/F]'))

# if st.button('Run Model'):
#     example = gpt_prompting.training_loop(data, option)
#     st.text(example)

