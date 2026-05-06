def seq_ratio(word1, word2):
    """
    Returns sequence match ratio for two words
    """
    raw_ratio = SequenceMatcher(None, word1, word2).ratio()
    return int(round(100 * raw_ratio))