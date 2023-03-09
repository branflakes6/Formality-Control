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

    path = kr_train + 'telephony.en-ko.formal.ko'
    data = dataframe_builder(data, path, 'Formal_Korean')
    path = kr_train + 'topical_chat.en-ko.formal.ko'
    data = dataframe_builder(data, path, 'Formal_Korean')

    path = kr_train + 'telephony.en-ko.informal.ko'
    data = dataframe_builder(data, path, 'Informal_Korean')
    path = kr_train + 'topical_chat.en-ko.informal.ko'
    data = dataframe_builder(data, path, 'Informal_Korean')

    data = pd.DataFrame(data)
    data.to_csv('./data/train/en-ko/en-kr_combined')
    return 0


def main():
    data_processing()
    data = pd.read_csv('data/train/en-ko/en-kr_combined_annotated')
    print(data['English'])


if __name__ == '__main__':
    main()
