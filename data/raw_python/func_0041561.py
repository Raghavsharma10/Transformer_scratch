def compare(stanzas, gold_schemes, found_schemes):
    """get accuracy and precision/recall"""
    result = SuccessMeasure()
    total = float(len(gold_schemes))
    correct = 0.0
    for (g, f) in zip(gold_schemes, found_schemes):
        if g == f:
            correct += 1
    result.accuracy = correct / total

    # for each word, let rhymeset[word] = set of words in rest of stanza rhyming with the word
    # precision = # correct words in rhymeset[word]/# words in proposed rhymeset[word]
    # recall = # correct words in rhymeset[word]/# words in reference words in rhymeset[word]
    # total precision and recall = avg over all words over all stanzas

    tot_p = 0.0
    tot_r = 0.0
    tot_words = 0.0

    for (s, g, f) in zip(stanzas, gold_schemes, found_schemes):
        stanzasize = len(s)
        for wi, word in enumerate(s):
            grhymeset_word = set(
                map(lambda x: x[0], filter(lambda x: x[1] == g[wi], zip(range(wi + 1, stanzasize), g[wi + 1:]))))
            frhymeset_word = set(
                map(lambda x: x[0], filter(lambda x: x[1] == f[wi], zip(range(wi + 1, stanzasize), f[wi + 1:]))))

            if len(grhymeset_word) == 0:
                continue

            tot_words += 1

            if len(frhymeset_word) == 0:
                continue

            # find intersection
            correct = float(len(grhymeset_word.intersection(frhymeset_word)))
            precision = correct / len(frhymeset_word)
            recall = correct / len(grhymeset_word)
            tot_p += precision
            tot_r += recall

    precision = tot_p / tot_words
    recall = tot_r / tot_words
    result.precision = precision
    result.recall = recall
    if precision + recall > 0:
        result.f_score = 2 * precision * recall / (precision + recall)
    return result