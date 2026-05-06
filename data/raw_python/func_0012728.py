def key2elements(key):
    """split key to elements"""
    # words = key.split('.')
    # if len(words) == 4:
    #     return words
    # # there is a dot in object name
    # fieldword = words.pop(-1)
    # nameword = '.'.join(words[-2:])
    # if nameword[-1] in ('"', "'"):
    #     # The object name is in quotes
    #     nameword = nameword[1:-1]
    # elements = words[:-2] + [nameword, fieldword, ]
    # return elements
    words = key.split('.')
    first2words = words[:2]
    lastword = words[-1]
    namewords = words[2:-1]
    namephrase = '.'.join(namewords)
    if namephrase.startswith("'") and namephrase.endswith("'"):
        namephrase = namephrase[1:-1]
    return first2words + [namephrase] + [lastword]