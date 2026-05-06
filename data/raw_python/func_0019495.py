def filter_significant_motifs(fname, result, bg, metrics=None):
    """Filter significant motifs based on several statistics.

    Parameters
    ----------
    fname : str
        Filename of output file were significant motifs will be saved.

    result : PredictionResult instance
        Contains motifs and associated statistics.

    bg : str
        Name of background type to use.

    metrics : sequence
        Metric with associated minimum values. The default is
        (("max_enrichment", 3), ("roc_auc", 0.55), ("enr_at_f[r", 0.55))

    Returns
    -------
    motifs : list
        List of Motif instances.
    """
    sig_motifs = []
    with open(fname, "w") as f:
        for motif in result.motifs:
            stats = result.stats.get(
                    "%s_%s" % (motif.id, motif.to_consensus()), {}).get(bg, {}
                    ) 
            if _is_significant(stats, metrics):
                f.write("%s\n" % motif.to_pfm())
                sig_motifs.append(motif)
    
    logger.info("%s motifs are significant", len(sig_motifs))
    logger.debug("written to %s", fname)
    
    return sig_motifs