def _read_motifs_from_filehandle(handle, fmt):
    """ 
    Read motifs from a file-like object.

    Parameters
    ----------
    handle : file-like object
        Motifs.
    fmt : string, optional
        Motif format, can be 'pwm', 'transfac', 'xxmotif', 'jaspar' or 'align'.
    
    Returns
    -------
    motifs : list
        List of Motif instances. 
    """
    if fmt.lower() == "pwm":
        motifs = _read_motifs_pwm(handle)
    if fmt.lower() == "transfac":
        motifs = _read_motifs_transfac(handle)
    if fmt.lower() == "xxmotif":
        motifs = _read_motifs_xxmotif(handle)
    if fmt.lower() == "align":
        motifs = _read_motifs_align(handle)
    if fmt.lower() == "jaspar":
        motifs = _read_motifs_jaspar(handle)
    
    if handle.name:
        base = os.path.splitext(handle.name)[0]
        map_file = base + ".motif2factors.txt"
        if os.path.exists(map_file):
            m2f_direct = {}
            m2f_indirect = {}
            for line in open(map_file):
                try:
                    motif,*factor_info = line.strip().split("\t")
                    if len(factor_info) == 1:
                        m2f_direct[motif] = factor_info[0].split(",")
                    elif len(factor_info) == 3:
                        if factor_info[2] == "Y":
                            m2f_direct[motif] = m2f_direct.get(motif, []) + [factor_info[0]]
                        else:
                            m2f_indirect[motif] = m2f_indirect.get(motif, []) + [factor_info[0]]
                except:
                    pass
            for motif in motifs:
                if motif.id in m2f_direct:
                    motif.factors[DIRECT_NAME] = m2f_direct[motif.id]
                if motif.id in m2f_indirect:
                    motif.factors[INDIRECT_NAME] = m2f_indirect[motif.id]
        for motif in motifs:
            for n in [DIRECT_NAME, INDIRECT_NAME]:
                motif.factors[n] = list(set(motif.factors[n]))
    return motifs