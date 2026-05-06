def expectation_step(t_table, stanzas, schemes, rprobs):
    """
     Compute posterior probability of schemes for each stanza
    """
    probs = numpy.zeros((len(stanzas), schemes.num_schemes))
    for i, stanza in enumerate(stanzas):
        scheme_indices = schemes.get_schemes_for_len(len(stanza))
        for scheme_index in scheme_indices:
            scheme = schemes.scheme_list[scheme_index]
            probs[i, scheme_index] = post_prob_scheme(t_table, stanza, scheme)
    probs = numpy.dot(probs, numpy.diag(rprobs))

    # Normalize
    scheme_sums = numpy.sum(probs, axis=1)
    for i, scheme_sum in enumerate(scheme_sums.tolist()):
        if scheme_sum > 0:
            probs[i, :] /= scheme_sum
    return probs