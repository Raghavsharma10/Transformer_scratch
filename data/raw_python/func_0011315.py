def variations(word):
    """Create variations of the word based on letter combinations like oo,
sh, etc."""

    if len(word) == 1:
        return [[word[0]]]
    elif word == 'aa':
        return [['A']]
    elif word == 'ee':
        return [['i']]
    elif word == 'ei':
        return [['ei']]
    elif word in ['oo', 'ou']:
        return [['u']]
    elif word == 'kha':
        return [['kha'], ['kh', 'a']]
    elif word in ['kh', 'gh', 'ch', 'sh', 'zh', 'ck']:
        return [[word]]
    elif word in ["'ee", "'ei"]:
        return [["'i"]]
    elif word in ["'oo", "'ou"]:
        return [["'u"]]
    elif word in ["a'", "e'", "o'", "i'", "u'", "A'"]:
        return [[word[0] + "'"]]
    elif word in ["'a", "'e", "'o", "'i", "'u", "'A"]:
        return [["'" + word[1]]]
    elif len(word) == 2 and word[0] == word[1]:
        return [[word[0]]]

    if word[:2] == 'aa':
        return [['A'] + i for i in variations(word[2:])]
    elif word[:2] == 'ee':
        return [['i'] + i for i in variations(word[2:])]
    elif word[:2] in ['oo', 'ou']:
        return [['u'] + i for i in variations(word[2:])]
    elif word[:3] == 'kha':
        return \
            [['kha'] + i for i in variations(word[3:])] + \
            [['kh', 'a'] + i for i in variations(word[3:])] + \
            [['k', 'h', 'a'] + i for i in variations(word[3:])]
    elif word[:2] in ['kh', 'gh', 'ch', 'sh', 'zh', 'ck']:
        return \
            [[word[:2]] + i for i in variations(word[2:])] + \
            [[word[0]] + i for i in variations(word[1:])]
    elif word[:2] in ["a'", "e'", "o'", "i'", "u'", "A'"]:
        return [[word[:2]] + i for i in variations(word[2:])]
    elif word[:3] in ["'ee", "'ei"]:
        return [["'i"] + i for i in variations(word[3:])]
    elif word[:3] in ["'oo", "'ou"]:
        return [["'u"] + i for i in variations(word[3:])]
    elif word[:2] in ["'a", "'e", "'o", "'i", "'u", "'A"]:
        return [[word[:2]] + i for i in variations(word[2:])]
    elif len(word) >= 2 and word[0] == word[1]:
        return [[word[0]] + i for i in variations(word[2:])]
    else:
        return [[word[0]] + i for i in variations(word[1:])]