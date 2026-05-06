def mp_calc_stats(motifs, fg_fa, bg_fa, bg_name=None):
    """Parallel calculation of motif statistics."""
    try:
        stats = calc_stats(motifs, fg_fa, bg_fa, ncpus=1)
    except Exception as e:
        raise
        sys.stderr.write("ERROR: {}\n".format(str(e)))
        stats = {}

    if not bg_name:
        bg_name = "default"

    return bg_name, stats