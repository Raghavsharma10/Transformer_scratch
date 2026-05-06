def threshold(args):
    """Calculate motif score threshold for a given FPR."""
    if args.fpr < 0 or args.fpr > 1:
        print("Please specify a FPR between 0 and 1")
        sys.exit(1)

    motifs = read_motifs(args.pwmfile)
    
    s = Scanner()
    s.set_motifs(args.pwmfile)
    s.set_threshold(args.fpr, filename=args.inputfile)

    print("Motif\tScore\tCutoff")
    for motif in motifs:
        min_score = motif.pwm_min_score()
        max_score = motif.pwm_max_score()
        opt_score = s.threshold[motif.id]
        if opt_score is None:
            opt_score = motif.pwm_max_score()
        threshold = (opt_score - min_score) / (max_score - min_score)
        print("{0}\t{1}\t{2}".format(
                motif.id, opt_score, threshold))