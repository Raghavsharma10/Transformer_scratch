def rank_motifs(stats, metrics=("roc_auc", "recall_at_fdr")):
    """Determine mean rank of motifs based on metrics."""
    rank = {}
    combined_metrics = []
    motif_ids = stats.keys()
    background = list(stats.values())[0].keys()
    for metric in metrics:
        mean_metric_stats = [np.mean(
            [stats[m][bg][metric] for bg in background]) for m in motif_ids]
        ranked_metric_stats = rankdata(mean_metric_stats)
        combined_metrics.append(ranked_metric_stats)
    
    for motif, val in zip(motif_ids, np.mean(combined_metrics, 0)):
        rank[motif] = val

    return rank