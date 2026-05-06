def align_two_alignments(aln1_fp, aln2_fp, moltype, params=None):
    """Returns an Alignment object from two existing Alignments.

    Parameters
    ----------
    aln1_fp : string
        file path of 1st alignment
    aln2_fp : string
        file path of 2nd alignment
    params : dict of parameters to pass in to the Mafft app controller.

    Returns
    -------
        The aligned sequences.
    """

    # Create Mafft app.
    app = Mafft(InputHandler='_input_as_paths',
                params=params,
                SuppressStderr=False)
    app._command = 'mafft-profile'

    # Get results using int_map as input to app
    res = app([aln1_fp, aln2_fp])

    return Alignment.read(res['StdOut'], constructor=moltype)