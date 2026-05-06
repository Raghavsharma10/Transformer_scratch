def get_num_words(text):
    """
    Counts and returns the number of words found in a given text

    :param text:
    :return:
    """
    try:
        word_regexp_pattern = re.compile(r"[a-zA-Záéíóúñ]+")
        num_words = re.findall(word_regexp_pattern, text)
        return len(num_words)
    except TypeError:
        return 0