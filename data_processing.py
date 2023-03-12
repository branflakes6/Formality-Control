import pandas as pd


def dataframe_builder(data, path, col):
    f = open(path, "r", encoding='utf-8')
    for line in f:
        data[col].append(line)
    return data

def data_processing_vt():
    kr_train = 'data/train/en-vi/formality-control.train.'
    data = {'English': [], 'Formal_Viet': [], 'Informal_Viet': []}

    path = kr_train + 'telephony.en-vi.en'
    data = dataframe_builder(data, path, "English")
    path = kr_train + 'topical_chat.en-vi.en'
    data = dataframe_builder(data, path, 'English')

    path = kr_train + 'telephony.en-vi.formal.vi'
    data = dataframe_builder(data, path, 'Formal_Viet')
    path = kr_train + 'topical_chat.en-vi.formal.vi'
    data = dataframe_builder(data, path, 'Formal_Viet')

    path = kr_train + 'telephony.en-vi.informal.vi'
    data = dataframe_builder(data, path, 'Informal_Viet')
    path = kr_train + 'topical_chat.en-vi.informal.vi'
    data = dataframe_builder(data, path, 'Informal_Viet')

    data = pd.DataFrame(data)
    data.to_csv('./data/train/en-vi/en-vi_combined')

def data_processing_vt_an():
    kr_train = 'data/train/en-vi/formality-control.train.'
    data = {'English': [], 'Formal_Viet': [], 'Informal_Viet': []}

    path = kr_train + 'telephony.en-vi.en'
    data = dataframe_builder(data, path, "English")
    path = kr_train + 'topical_chat.en-vi.en'
    data = dataframe_builder(data, path, 'English')

    path = kr_train + 'telephony.en-vi.formal.annotated.vi'
    data = dataframe_builder(data, path, 'Formal_Viet')
    path = kr_train + 'topical_chat.en-vi.formal.annotated.vi'
    data = dataframe_builder(data, path, 'Formal_Viet')

    path = kr_train + 'telephony.en-vi.informal.annotated.vi'
    data = dataframe_builder(data, path, 'Informal_Viet')
    path = kr_train + 'topical_chat.en-vi.informal.annotated.vi'
    data = dataframe_builder(data, path, 'Informal_Viet')

    data = pd.DataFrame(data)
    data.to_csv('./data/train/en-vi/en-vi_combined-annotated')


def data_processing_kr():
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
    data.to_csv('./data/train/en-ko/en-kr_combined')


def data_processing_kr_an():
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


def main():
    data_processing_vt_an()


if __name__ == '__main__':
    main()
