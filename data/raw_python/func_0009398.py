def homogeneity(transition_matrices, regime_names=[], class_names=[],
                title="Markov Homogeneity Test"):
    """
    Test for homogeneity of Markov transition probabilities across regimes.

    Parameters
    ----------
    transition_matrices : list
                          of transition matrices for regimes, all matrices must
                          have same size (r, c). r is the number of rows in the
                          transition matrix and c is the number of columns in
                          the transition matrix.
    regime_names        : sequence
                          Labels for the regimes.
    class_names         : sequence
                          Labels for the classes/states of the Markov chain.
    title               : string
                          name of test.

    Returns
    -------
                        : implicit
                          an instance of Homogeneity_Results.
    """

    return Homogeneity_Results(transition_matrices, regime_names=regime_names,
                               class_names=class_names, title=title)