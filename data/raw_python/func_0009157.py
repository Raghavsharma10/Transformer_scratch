def _load_data():
    """Load the word and character mapping data into a dictionary.

    In the data files, each line is formatted like this:
        HANZI   PINYIN_READING/PINYIN_READING

    So, lines need to be split by '\t' and then the Pinyin readings need to be
    split by '/'.

    """
    data = {}
    for name, file_name in (('words', 'hanzi_pinyin_words.tsv'),
                            ('characters', 'hanzi_pinyin_characters.tsv')):
        # Split the lines by tabs: [[hanzi, pinyin]...].
        lines = [line.split('\t') for line in
                 dragonmapper.data.load_data_file(file_name)]
        # Make a dictionary: {hanzi: [pinyin, pinyin]...}.
        data[name] = {hanzi: pinyin.split('/') for hanzi, pinyin in lines}
    return data