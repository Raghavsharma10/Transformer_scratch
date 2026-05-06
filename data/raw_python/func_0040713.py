def iterate(t_table, wordlist, stanzas, schemes, rprobs, maxsteps):
    """
    Iterate EM and return final probabilities
    """
    data_probs = numpy.zeros(len(stanzas))
    old_data_probs = None
    probs = None
    num_words = len(wordlist)

    ctr = 0
    for ctr in range(maxsteps):
        logging.info("Iteration {}".format(ctr))
        old_data_probs = data_probs

        logging.info("Expectation step")
        probs = expectation_step(t_table, stanzas, schemes, rprobs)

        logging.info("Maximization step")
        t_table, rprobs = maximization_step(num_words, stanzas, schemes, probs)

    # Warn if it did not converge
    data_probs = numpy.logaddexp.reduce(probs, axis=1)
    if ctr == maxsteps - 1 and not numpy.allclose(data_probs, old_data_probs):
        logging.warning("Warning: EM did not converge")

    logging.info("Stopped after {} interations".format(ctr))
    return probs