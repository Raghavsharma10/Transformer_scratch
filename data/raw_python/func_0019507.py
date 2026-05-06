def calc_stats_iterator(motifs, fg_file, bg_file, genome=None, stats=None, ncpus=None):
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
    if not stats:
        stats = rocmetrics.__all__
    
    if isinstance(motifs, Motif):
        all_motifs = [motifs]
    else:
        if type([]) == type(motifs):
            all_motifs = motifs
        else:
            all_motifs = read_motifs(motifs, fmt="pwm")
    
    if ncpus is None:
        ncpus = int(MotifConfig().get_default_params()["ncpus"])
    chunksize = 240

    for i in range(0, len(all_motifs), chunksize):
        result = {}
        logger.debug("chunk %s of %s",
            (i / chunksize) + 1, len(all_motifs) // chunksize + 1)
        motifs = all_motifs[i:i + chunksize]
       
        fg_total = scan_to_best_match(fg_file, motifs, ncpus=ncpus, genome=genome)
        bg_total = scan_to_best_match(bg_file, motifs, ncpus=ncpus, genome=genome)
     
        logger.debug("calculating statistics")
        
        if ncpus == 1:
            it = _single_stats(motifs, stats, fg_total, bg_total) 
        else:
            it = _mp_stats(motifs, stats, fg_total, bg_total, ncpus) 
        
        for motif_id, s, ret in it:
            if motif_id not in result:
                result[motif_id] = {}
            result[motif_id][s] = ret
        yield result