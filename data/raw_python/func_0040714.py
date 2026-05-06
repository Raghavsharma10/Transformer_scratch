def get_results(probs, stanzas, schemes):
    """
    Returns a list of tuples (
        stanza [as list of final words],
        best scheme [as list of integers]
    )
    """
    results = []
    for i, stanza in enumerate(stanzas):
        best_scheme = schemes.scheme_list[numpy.argmax(probs[i, :])]
        results.append((stanza.words, best_scheme))
    return results