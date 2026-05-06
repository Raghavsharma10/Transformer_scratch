def rename_motifs(motifs, stats=None):
    """Rename motifs to GimmeMotifs_1..GimmeMotifs_N.
    
    If stats object is passed, stats will be copied."""
    final_motifs = []
    for i, motif in enumerate(motifs):
        old = str(motif)
        motif.id = "GimmeMotifs_{}".format(i + 1)
        final_motifs.append(motif)
        if stats:
            stats[str(motif)] = stats[old].copy()
    
    if stats:
        return final_motifs, stats
    else:
        return final_motifs