def censor(input_text):
    """ Returns the input string with profanity replaced with a random string
    of characters plucked from the censor_characters pool.

    """
    ret = input_text
    words = get_words()
    for word in words:
        curse_word = re.compile(re.escape(word), re.IGNORECASE)
        cen = "".join(get_censor_char() for i in list(word))
        ret = curse_word.sub(cen, ret)
    return ret