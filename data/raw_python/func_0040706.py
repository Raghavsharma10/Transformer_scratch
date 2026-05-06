def get_rhymelists(stanza, scheme):
    """
    Returns ordered lists of the stanza's word indices as defined by given scheme
    """
    rhymelists = defaultdict(list)
    for rhyme_group, word_index in zip(scheme, stanza.word_indices):
        rhymelists[rhyme_group].append(word_index)
    return list(rhymelists.values())