def default_motifs():
    """Return list of Motif instances from default motif database."""
    config = MotifConfig()
    d = config.get_motif_dir()
    m = config.get_default_params()['motif_db']

    if not d or not m:
        raise ValueError("default motif database not configured")

    fname = os.path.join(d, m)
    with open(fname) as f:
        motifs = read_motifs(f)
    
    return motifs