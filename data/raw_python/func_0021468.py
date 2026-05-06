def rnd_ins_seq(ins_len, C_R, CP_first_nt):
    """Generate a random insertion nucleotide sequence of length ins_len.

    Draws the sequence identity (for a set length) from the distribution
    defined by the dinucleotide markov model of transition matrix R.

    Parameters
    ----------
    ins_len : int
        Length of nucleotide sequence to be inserted.
    C_R : ndarray
        (4, 4) array of the cumulative transition probabilities defined by the
        Markov transition matrix R
    CP_first_nt : ndarray
        (4,) array of the cumulative probabilities for the first inserted
        nucleotide

    Returns
    -------
    seq : str
        Randomly generated insertion sequence of length ins_len.

    Examples
    --------
    >>> rnd_ins_seq(7, CP_generative_model['C_Rvd'], CP_generative_model['C_first_nt_bias_insVD'])
    'GATGGAC'
    >>> rnd_ins_seq(7, CP_generative_model['C_Rvd'], CP_generative_model['C_first_nt_bias_insVD'])
    'ACCCCCG'
    >>> rnd_ins_seq(3, CP_generative_model['C_Rvd'], CP_generative_model['C_first_nt_bias_insVD'])
    'GCC'

    """
    nt2num = {'A': 0, 'C': 1, 'G': 2, 'T': 3}
    num2nt = 'ACGT'

    if ins_len == 0:
        return ''

    seq = num2nt[CP_first_nt.searchsorted(np.random.random())]
    ins_len += -1

    while ins_len > 0:
        seq += num2nt[C_R[nt2num[seq[-1]], :].searchsorted(np.random.random())]
        ins_len += -1

    return seq