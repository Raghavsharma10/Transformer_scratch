def calc_stats(motifs, fg_file, bg_file, genome=None, stats=None, ncpus=None):
    """Calculate motif enrichment metrics.

    Parameters
    ----------
    motifs : str, list or Motif instance
        A file with motifs in pwm format, a list of Motif instances or a 
        single Motif instance.

    fg_file : str
        Filename of a FASTA, BED or region file with positive sequences.

    bg_file : str
        Filename of a FASTA, BED or region file with negative sequences.

    genome : str, optional
        Genome or index directory in case of BED/regions.
    
    stats : list, optional
        Names of metrics to calculate. See gimmemotifs.rocmetrics.__all__ 
        for available metrics.

    ncpus : int, optional
        Number of cores to use.

    Returns
    -------
    result : dict
        Dictionary with results where keys are motif ids and the values are
        dictionary with metric name and value pairs.
    """
    result = {}
    for batch_result in calc_stats_iterator(motifs, fg_file, bg_file, genome=genome, stats=stats, ncpus=ncpus):
        for motif_id in batch_result:
            if motif_id not in result:
                result[motif_id] = {}
            for s,ret in batch_result[motif_id].items():
                result[motif_id][s] = ret
    return result