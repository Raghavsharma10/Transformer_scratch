def create_roc_plots(pwmfile, fgfa, background, outdir):
    """Make ROC plots for all motifs."""
    motifs = read_motifs(pwmfile, fmt="pwm", as_dict=True)
    ncpus = int(MotifConfig().get_default_params()['ncpus'])
    pool = Pool(processes=ncpus)
    jobs = {}
    for bg,fname in background.items():
        for m_id, m in motifs.items():

            k = "{}_{}".format(str(m), bg)
            jobs[k] = pool.apply_async(
                                            get_roc_values,
                                            (motifs[m_id], fgfa, fname,)
                                            )
    imgdir = os.path.join(outdir, "images")
    if not os.path.exists(imgdir):
        os.mkdir(imgdir)
    
    roc_img_file = os.path.join(outdir, "images", "{}_roc.{}.png")

    for motif in motifs.values():
        for bg in background:
            k = "{}_{}".format(str(motif), bg)
            error, x, y = jobs[k].get()
            if error:
                logger.error("Error in thread: %s", error)
                logger.error("Motif: %s", motif)
                sys.exit(1)
            roc_plot(roc_img_file.format(motif.id, bg), x, y)