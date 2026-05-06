def get_wordlist(stanzas):
    """
    Get an iterable of all final words in all stanzas
    """
    return sorted(list(set().union(*[stanza.words for stanza in stanzas])))