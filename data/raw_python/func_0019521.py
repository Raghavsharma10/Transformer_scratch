def scan_to_best_match(fname, motifs, ncpus=None, genome=None, score=False):
    """Scan a FASTA file with motifs.

    Scan a FASTA file and return a dictionary with the best match per motif.

    Parameters
    ----------
    fname : str
        Filename of a sequence file in FASTA format.

    motifs : list
        List of motif instances.

    Returns
    -------
    result : dict
        Dictionary with motif scanning results.
    """
    # Initialize scanner
    s = Scanner(ncpus=ncpus)
    s.set_motifs(motifs)
    s.set_threshold(threshold=0.0)
    if genome:
        s.set_genome(genome)
    
    if isinstance(motifs, six.string_types):
        motifs = read_motifs(motifs)

    logger.debug("scanning %s...", fname)
    result = dict([(m.id, []) for m in motifs])
    if score:
        it = s.best_score(fname)
    else:
        it = s.best_match(fname)
    for scores in it:
        for motif,score in zip(motifs, scores):
            result[motif.id].append(score)

    # Close the pool and reclaim memory
    del s
    
    return result