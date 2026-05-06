def add_seqs_to_alignment(seqs_fp, aln_fp, moltype=DNA,
                          params=None, accurate=False):
    """Returns an Alignment object from seqs and existing Alignment.

    The "--seed" option can be used for adding unaligned sequences into
    a highly reliable alignment (seed) consisting of a small number of
    sequences.

    Parameters
    ----------
    seqs_fp : string
        file path of the unaligned sequences
    aln_fp : string
        file path of the seed alignment
    params : dict of parameters to pass in to the Mafft app controller.

    Returns
    -------
        The aligned sequences. The seq in the seed alignment will have
        "_seed_" prefixed to their seq id.
    """
    if params is None:
        params = {'--seed': aln_fp}
    else:
        params['--seed'] = aln_fp

    return align_unaligned_seqs(seqs_fp, moltype, params, accurate)