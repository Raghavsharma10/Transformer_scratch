def count_tf(tokens_stream):
    """
    Count term frequencies for a single file.
    """
    tf = defaultdict(int)
    for tokens in tokens_stream:
        for token in tokens:
            tf[token] += 1
    return tf