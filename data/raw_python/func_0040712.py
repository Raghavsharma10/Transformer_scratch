def maximization_step(num_words, stanzas, schemes, probs):
    """
    Update latent variables t_table, rprobs
    """
    t_table = numpy.zeros((num_words, num_words + 1))
    rprobs = numpy.ones(schemes.num_schemes)
    for i, stanza in enumerate(stanzas):
        scheme_indices = schemes.get_schemes_for_len(len(stanza))
        for scheme_index in scheme_indices:
            myprob = probs[i, scheme_index]
            rprobs[scheme_index] += myprob
            scheme = schemes.scheme_list[scheme_index]
            rhymelists = get_rhymelists(stanza, scheme)
            for rhymelist in rhymelists:
                for j, word_index in enumerate(rhymelist):
                    t_table[word_index, -1] += myprob
                    for word_index2 in rhymelist[:j] + rhymelist[j + 1:]:
                        t_table[word_index, word_index2] += myprob

    # Normalize t_table
    t_table_sums = numpy.sum(t_table, axis=0)
    for i, t_table_sum in enumerate(t_table_sums.tolist()):
        if t_table_sum != 0:
            t_table[:, i] /= t_table_sum

    # Normalize rprobs
    totrprob = numpy.sum(rprobs)
    rprobs /= totrprob

    return t_table, rprobs