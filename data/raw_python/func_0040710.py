def post_prob_scheme(t_table, stanza, scheme):
    """
    Compute posterior probability of a scheme for a stanza, with probability of every word in rhymelist
    rhyming with all the ones before it
    """
    myprob = 1
    rhymelists = get_rhymelists(stanza, scheme)
    for rhymelist in rhymelists:
        for i, word_index in enumerate(rhymelist):
            if i == 0:  # first word, use P(w|x)
                myprob *= t_table[word_index, -1]
            else:
                for word_index2 in rhymelist[:i]:  # history
                    myprob *= t_table[word_index, word_index2]
    if myprob == 0 and len(stanza) > 30:  # probably underflow
        myprob = 1e-300
    return myprob