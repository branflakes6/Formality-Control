from langchain.prompts import PromptTemplate
from langchain.llms import OpenAI
from langchain.chains import LLMChain
import os
from dotenv import load_dotenv
load_dotenv()

OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
LLM = OpenAI(temperature = 0, model_name= "text-davinci-003")
# prompt = PromptTemplate(
#     input_variables=["sentence"],
#     #template="Come up with 10 unique and interesting names for a baby girl born in {country}.",
#     template="English: Yeah Did your mom know you were throwing the party?\n Formal Korean: 그, 어머님은 [F]님이[/F] 그 파티 연 거 [F]아세요[/F]?\n Informal Korean: 그, 어머님은 [F]네가[/F] 그 파티 연 거 [F]아셔[/F]?\n English: [F]내가[/F] [F]너[/F] 잘못되라고 하고 있다는 [F]거야[/F]?\n Formal Korean:[F]제가[/F] [F]님[/F] 잘못되라고 하고 있다는 [F]거예요[/F]?\n Informal Korean:[F]내가[/F] [F]너[/F] 잘못되라고 하고 있다는 [F]거야[/F]?\n English: {sentence} "
# )

# chain = LLMChain(llm=LLM, prompt=prompt)
# chain.run("That means that you don't have enough space on your computer for some reason.")
#무슨 이유에서든 컴퓨터에 공간이 충분하지 않다는 [F]뜻이에요[/F].
#무슨 이유에서든 컴퓨터에 공간이 충분하지 않다는 [F]뜻이야[/F].
#Formal Output: 그러면 [F]님[/F]의 컴퓨터에 어떤 이유로 인해 충분한 공간이 없다는 걸 의미합니다
# English: Then it means that [F][/F]'s computer doesn't have enough space for some reason.

prompt = PromptTemplate(
    input_variables=["sentence"],
    #template="Come up with 10 unique and interesting names for a baby girl born in {country}.",
    template="Formal Korean: 그, 어머님은 [F]님이[/F] 그 파티 연 거 [F]아세요[/F]?\n Informal Korean: 그, 어머님은 [F]네가[/F] 그 파티 연 거 [F]아셔[/F]?\n Formal Korean:[F]제가[/F] [F]님[/F] 잘못되라고 하고 있다는 [F]거예요[/F]?\n Informal Korean:[F]내가[/F] [F]너[/F] 잘못되라고 하고 있다는 [F]거야[/F]?\n Formal Korean: {sentence} "
)

chain = LLMChain(llm=LLM, prompt=prompt)
print(chain.run("그게 [F]님이[/F] 제일 [F]좋아하시는[/F] [F]피자예요[/F]?"))
#그게 [F]네가[/F] 제일 [F]좋아하는[/F] [F]피자야[/F]?