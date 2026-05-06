def get_wordset(poems):
    """get all words"""
    words = sorted(list(set(reduce(lambda x, y: x + y, poems))))
    return words