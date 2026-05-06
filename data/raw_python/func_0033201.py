def align_unaligned_seqs(seqs_fp, moltype=DNA, params=None, accurate=False):
    """Aligns unaligned sequences

    Parameters
    ----------
    seqs_fp : string
        file path of the input fasta file
    moltype : {skbio.DNA, skbio.RNA, skbio.Protein}
    params : dict-like type
        It pass the additional parameter settings to the application.
        Default is None.
    accurate : boolean
        Perform accurate alignment or not. It will sacrifice performance
        if set to True. Default is False.

    Returns
    -------
    Alignment object
        The aligned sequences.

    See Also
    --------
    skbio.Alignment
    skbio.DNA
    skbio.RNA
    skbio.Protein
    """
    # Create Mafft app.
    app = Mafft(InputHandler='_input_as_path', params=params)

    # Turn on correct sequence type
    app.Parameters[MOLTYPE_MAP[moltype]].on()

    # Do not report progress
    app.Parameters['--quiet'].on()

    # More accurate alignment, sacrificing performance.
    if accurate:
        app.Parameters['--globalpair'].on()
        app.Parameters['--maxiterate'].Value = 1000

    # Get results using int_map as input to app
    res = app(seqs_fp)

    # Get alignment as dict out of results
    alignment = Alignment.read(res['StdOut'], constructor=moltype)

    # Clean up
    res.cleanUp()

    return alignment