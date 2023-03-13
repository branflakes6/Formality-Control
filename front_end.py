import streamlit as st
import time

st.title('Korean Formality Control')

def spoof(input):
    with st.spinner('Translating'):
        time.sleep(3)
    formal_out = "그래서 아직도 너무 좋아하는 미문에 대한 애정을 말해요. 또한 애정을 이해해야 합니다."
    with st.spinner('Converting'):
        time.sleep(3)
    time.sleep(3)
    informal_out = "그래서 너무 좋아하는 미문에 대한 이유를 말해봐. 내가 왜 그렇게 좋아하는지 이해해야지."
    return formal_out, informal_out



data = [
    'Formal Korean: 그, 어머님은 [F]님이[/F] 그 파티 연 거 [F]아세요[/F]?\n Informal Korean: 그, 어머님은 [F]네가[/F] 그 파티 연 거 [F]아셔[/F]?\n Formal Korean:[F]제가[/F] [F]님[/F] 잘못되라고 하고 있다는 [F]거예요[/F]?\n Informal Korean:[F]내가[/F] [F]너[/F] 잘못되라고 하고 있다는 [F]거야[/F]?\n Formal Korean: {sentence}']


input = st.text_input('Enter an English sentence to translate', '')
if st.button('Translate'):
    formal, informal = spoof(input)
    st.write("Formal Korean", formal)
    st.write("Informal Korean", informal)



