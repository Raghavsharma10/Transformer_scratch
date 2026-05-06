def extract_noun_phrases(tagged_doc):
    """
    (From textblob)
    """
    tags = _normalize_tags(tagged_doc)
    merge = True
    while merge:
        merge = False
        for x in range(0, len(tags) - 1):
            t1 = tags[x]
            t2 = tags[x + 1]
            key = t1[1], t2[1]
            value = CFG.get(key, '')
            if value:
                merge = True
                tags.pop(x)
                tags.pop(x)
                match = '%s %s' % (t1[0], t2[0])
                pos = value
                tags.insert(x, (match, pos))
                break

    matches = [t[0] for t in tags if t[1] in ['NNP', 'NNI']]
    return matches