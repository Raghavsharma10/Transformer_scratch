def best_motif_in_cluster(single_pwm, clus_pwm, clusters, fg_fa, background, stats=None, metrics=("roc_auc", "recall_at_fdr")):
    """Return the best motif per cluster for a clustering results.

    The motif can be either the average motif or one of the clustered motifs.

    Parameters
    ----------
    single_pwm : str
        Filename of motifs.

    clus_pwm : str
        Filename of motifs.

    clusters : 
        Motif clustering result.

    fg_fa : str
        Filename of FASTA file.

    background : dict
        Dictionary for background file names.

    stats : dict, optional
        If statistics are not supplied they will be computed.

    metrics : sequence, optional
        Metrics to use for motif evaluation. Default are "roc_auc" and 
        "recall_at_fdr".
    
    Returns
    -------
    motifs : list
        List of Motif instances.
    """
    # combine original and clustered motifs
    motifs = read_motifs(single_pwm) + read_motifs(clus_pwm)
    motifs = dict([(str(m), m) for m in motifs])

    # get the statistics for those motifs that were not yet checked
    clustered_motifs = []
    for clus,singles in clusters:
        for motif in set([clus] + singles):
            if str(motif) not in stats:
                clustered_motifs.append(motifs[str(motif)])
    
    new_stats = {}
    for bg, bg_fa in background.items():
        for m,s in calc_stats(clustered_motifs, fg_fa, bg_fa).items():
            if m not in new_stats:
                new_stats[m] = {}
            new_stats[m][bg] = s
    stats.update(new_stats)
    
    rank = rank_motifs(stats, metrics)

    # rank the motifs
    best_motifs = []
    for clus, singles in clusters:
        if len(singles) > 1:
            eval_motifs = singles
            if clus not in motifs:
                eval_motifs.append(clus)
            eval_motifs = [motifs[str(e)] for e in eval_motifs]
            best_motif = sorted(eval_motifs, key=lambda x: rank[str(x)])[-1]
            best_motifs.append(best_motif)
        else:
            best_motifs.append(clus)
        for bg in background:
            stats[str(best_motifs[-1])][bg]["num_cluster"] = len(singles)

    best_motifs = sorted(best_motifs, key=lambda x: rank[str(x)], reverse=True)
    return best_motifs