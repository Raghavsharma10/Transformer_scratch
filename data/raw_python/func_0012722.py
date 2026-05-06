def _word_badness(word):
    """
    Assign a heuristic to possible outputs from Morphy. Minimizing this
    heuristic avoids incorrect stems.
    """
    if word.endswith('e'):
        return len(word) - 2
    elif word.endswith('ess'):
        return len(word) - 10
    elif word.endswith('ss'):
        return len(word) - 4
    else:
        return len(word)