def create_denovo_motif_report(inputfile, pwmfile, fgfa, background, locfa, outdir, params, stats=None):
    """Create text and graphical (.html) motif reports."""
    logger.info("creating reports")

    motifs = read_motifs(pwmfile, fmt="pwm")
    
    # ROC plots
    create_roc_plots(pwmfile, fgfa, background, outdir)
    
    # Closest match in database
    mc = MotifComparer()
    closest_match = mc.get_closest_match(motifs)
    
    if stats is None:
        stats = {}
        for bg, bgfa in background.items():
            for m, s in calc_stats(motifs, fgfa, bgfa).items():
                if m not in stats:
                    stats[m] = {}
                stats[m][bg] = s

    stats = add_star(stats)

    if not params:
        params = {}
    cutoff_fpr = params.get('cutoff_fpr', 0.9)
    lwidth = np.median([len(seq) for seq in Fasta(locfa).seqs])

    # Location plots
    logger.debug("Creating localization plots")
    for motif in motifs:
        logger.debug("  {} {}".format(motif.id, motif))
        outfile = os.path.join(outdir, "images/{}_histogram.svg".format(motif.id))
        motif_localization(locfa, motif, lwidth, outfile, cutoff=cutoff_fpr)

    # Create reports
    _create_text_report(inputfile, motifs, closest_match, stats, outdir)
    _create_graphical_report(inputfile, pwmfile, background, closest_match, outdir, stats)