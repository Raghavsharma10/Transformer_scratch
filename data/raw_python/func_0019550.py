def read_motifs(infile=None, fmt="pwm", as_dict=False):
    """ 
    Read motifs from a file or stream or file-like object.

    Parameters
    ----------
    infile : string or file-like object, optional
        Motif database, filename of motif file or file-like object. If infile 
        is not specified the default motifs as specified in the config file 
        will be returned.

    fmt : string, optional
        Motif format, can be 'pwm', 'transfac', 'xxmotif', 'jaspar' or 'align'.
    
    as_dict : boolean, optional
        Return motifs as a dictionary with motif_id, motif pairs.
    
    Returns
    -------
    motifs : list
        List of Motif instances. If as_dict is set to True, motifs is a 
        dictionary.
    """
    if infile is None or isinstance(infile, six.string_types): 
        infile = pwmfile_location(infile)
        with open(infile) as f:
            motifs = _read_motifs_from_filehandle(f, fmt)
    else:
        motifs = _read_motifs_from_filehandle(infile, fmt)

    if as_dict:
        motifs = {m.id:m for m in motifs}

    return motifs