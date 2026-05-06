def parse_motifs(motifs):
    """Parse motifs in a variety of formats to return a list of motifs.

    Parameters
    ----------

    motifs : list or str
        Filename of motif,  list of motifs or single Motif instance.

    Returns
    -------

    motifs : list
        List of Motif instances.
    """
    if isinstance(motifs, six.string_types):
        with open(motifs) as f:
            if motifs.endswith("pwm") or motifs.endswith("pfm"):
                motifs = read_motifs(f, fmt="pwm")
            elif motifs.endswith("transfac"):
                motifs = read_motifs(f, fmt="transfac")
            else: 
                motifs = read_motifs(f)
    elif isinstance(motifs, Motif):
        motifs = [motifs]
    else:
        if not isinstance(list(motifs)[0], Motif):
            raise ValueError("Not a list of motifs")
    
    return list(motifs)