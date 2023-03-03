import pandas as pd


def dataframe_builder(data, path, col):
    f = open(path, "r", encoding='utf-8')
    for line in f:
        data[col].append(line)
    return data


def data_processing():
    kr_train = 'data/train/en-ko/formality-control.train.'
    data = {'English': [], 'Formal_Korean': [], 'Informal_Korean': []}

    path = kr_train + 'telephony.en-ko.en'
    data = dataframe_builder(data, path, "English")
    path = kr_train + 'topical_chat.en-ko.en'
    data = dataframe_builder(data, path, 'English')

    path = kr_train + 'telephony.en-ko.formal.annotated.ko'
    data = dataframe_builder(data, path, 'Formal_Korean')
    path = kr_train + 'topical_chat.en-ko.formal.annotated.ko'
    data = dataframe_builder(data, path, 'Formal_Korean')

    path = kr_train + 'telephony.en-ko.informal.annotated.ko'
    data = dataframe_builder(data, path, 'Informal_Korean')
    path = kr_train + 'topical_chat.en-ko.informal.annotated.ko'
    data = dataframe_builder(data, path, 'Informal_Korean')

    data = pd.DataFrame(data)
    data.to_csv('./data/train/en-ko/en-kr_combined_annotated')
    return 0


# Loop through n rows of the data
# Create a string in the form 'English: english, Formal Korean: formal, Informal Korean: inforaml'
# Add all strings together
# Ensure variable is not in the prompt (Cant have the sentence we are translating in our prompt or model will know the answer)
def english_to_formal_informal_prompt(data, n, variable):
    return 0


# Prompt 'Formal Korean: formal, Informal Korean: informal'
def formal_to_formal_prompt(data, n, variable):
    return 0


def main():
    # data_processing()
    data = pd.read_csv('data/train/en-ko/en-kr_combined_annotated')
    print(data['English'])


if __name__ == '__main__':
    main()
