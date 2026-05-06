def roc(args):
    """ Calculate ROC_AUC and other metrics and optionally plot ROC curve."""
    outputfile = args.outfile
    # Default extension for image
    if outputfile and not outputfile.endswith(".png"):
        outputfile += ".png"
    
    motifs = read_motifs(args.pwmfile, fmt="pwm")

    ids = []
    if args.ids:
        ids = args.ids.split(",")
    else:
        ids = [m.id for m in motifs]
    motifs = [m for m in motifs if (m.id in ids)]
    
    stats = [
            "phyper_at_fpr",
            "roc_auc", 
            "pr_auc", 
            "enr_at_fpr",
            "recall_at_fdr", 
            "roc_values",
            "matches_at_fpr",
            ]
    
    plot_x = []
    plot_y = []
    legend = []
    
    f_out = sys.stdout
    if args.outdir:
        if not os.path.exists(args.outdir):
            os.makedirs(args.outdir)
        f_out = open(args.outdir + "/gimme.roc.report.txt", "w")
    
    # Print the metrics
    f_out.write("Motif\t# matches\t# matches background\tP-value\tlog10 P-value\tROC AUC\tPR AUC\tEnr. at 1% FPR\tRecall at 10% FDR\n")
    
    
    for motif_stats in calc_stats_iterator(motifs, args.sample, args.background, 
            genome=args.genome, stats=stats, ncpus=args.ncpus):
    
        for motif in motifs:
            if str(motif) in motif_stats:
                if outputfile:
                    x, y = motif_stats[str(motif)]["roc_values"]
                    plot_x.append(x)
                    plot_y.append(y)
                    legend.append(motif.id)
                log_pvalue = np.inf
                if motif_stats[str(motif)]["phyper_at_fpr"] > 0:
                    log_pvalue = -np.log10(motif_stats[str(motif)]["phyper_at_fpr"])
                f_out.write("{}\t{:d}\t{:d}\t{:.2e}\t{:.3f}\t{:.3f}\t{:.3f}\t{:.2f}\t{:0.4f}\n".format(
                      motif.id, 
                      motif_stats[str(motif)]["matches_at_fpr"][0], 
                      motif_stats[str(motif)]["matches_at_fpr"][1], 
                      motif_stats[str(motif)]["phyper_at_fpr"], 
                      log_pvalue, 
                      motif_stats[str(motif)]["roc_auc"], 
                      motif_stats[str(motif)]["pr_auc"], 
                      motif_stats[str(motif)]["enr_at_fpr"], 
                      motif_stats[str(motif)]["recall_at_fdr"],
                      ))
    f_out.close() 
    
    if args.outdir:
        html_report(
            args.outdir,
            args.outdir + "/gimme.roc.report.txt",
            args.pwmfile,
            0.01,
            )

    # Plot the ROC curve
    if outputfile:
        roc_plot(outputfile, plot_x, plot_y, ids=legend)