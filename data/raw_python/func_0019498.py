def gimme_motifs(inputfile, outdir, params=None, filter_significant=True, cluster=True, create_report=True):
    """De novo motif prediction based on an ensemble of different tools.

    Parameters
    ----------
    inputfile : str
        Filename of input. Can be either BED, narrowPeak or FASTA.

    outdir : str
        Name of output directory.

    params : dict, optional
        Optional parameters.

    filter_significant : bool, optional
        Filter motifs for significance using the validation set.
    
    cluster : bool, optional
        Cluster similar predicted (and significant) motifs.

    create_report : bool, optional
        Create output reports (both .txt and .html).
 
    Returns
    -------
    motifs : list
        List of predicted motifs.     

    Examples
    --------

    >>> from gimmemotifs.denovo import gimme_motifs
    >>> gimme_motifs("input.fa", "motifs.out")
    """
    if outdir is None:
        outdir = "gimmemotifs_{}".format(datetime.date.today().strftime("%d_%m_%Y"))

    # Create output directories
    tmpdir = os.path.join(outdir, "intermediate")
    for d in [outdir, tmpdir]: 
        if not os.path.exists(d):
            os.mkdir(d) 

    # setup logfile
    logger = logging.getLogger("gimme")
    # Log to file
    logfile = os.path.join(outdir, "gimmemotifs.log")
    fh = logging.FileHandler(logfile, "w")
    fh.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    fh.setFormatter(file_formatter)
    logger.addHandler(fh)
    logger = logging.getLogger("gimme.denovo.gimme_motifs")
    
    # Initialize parameters
    params = parse_denovo_params(params)
 
    # Check the input files
    input_type, background = check_denovo_input(inputfile, params)
   
    logger.info("starting full motif analysis")
    logger.debug("Using temporary directory %s", mytmpdir())
    
    
    # Create the necessary files for motif prediction and validation
    if input_type == "bed":
        prepare_denovo_input_bed(inputfile, params, tmpdir)
    elif input_type == "narrowpeak":
        prepare_denovo_input_narrowpeak(inputfile, params, tmpdir)
    elif input_type == "fasta":
        prepare_denovo_input_fa(inputfile, params, tmpdir)
    else:
        
        logger.error("Unknown input file.")
        sys.exit(1)

    # Create the background FASTA files
    background = create_backgrounds(
            tmpdir, 
            background, 
            params.get("genome", None), 
            params["width"], 
            params.get("custom_background", None)
            )
    
    # Predict de novo motifs
    result = predict_motifs(
            os.path.join(tmpdir, "prediction.fa"),
            os.path.join(tmpdir, "prediction.bg.fa"),
            os.path.join(tmpdir, "all_motifs.pfm"),
            params=params,
            stats_fg=os.path.join(tmpdir, 'validation.fa'),
            stats_bg=background, 
            )

    if len(result.motifs) == 0:
        logger.info("finished")
        return []
    
    # Write statistics
    stats_file = os.path.join(tmpdir, "stats.{}.txt")
    write_stats(result.stats, stats_file)

    bg = sorted(background, key=lambda x: BG_RANK[x])[0]
    if filter_significant:
        motifs = filter_significant_motifs(
                os.path.join(tmpdir, "significant_motifs.pfm"),
                result, 
                bg)
        if len(motifs) == 0:
            logger.info("no significant motifs")
            return 

        pwmfile = os.path.join(tmpdir, "significant_motifs.pfm")
    else:
        logger.info("not filtering for significance")
        motifs = result.motifs
        pwmfile = os.path.join(tmpdir, "all_motifs.pfm")

    if cluster: 
        clusters = cluster_motifs_with_report(
                    pwmfile,
                    os.path.join(tmpdir, "clustered_motifs.pfm"),
                    outdir,
                    0.95,
                    title=inputfile)
        
        # Determine best motif in cluster
        best_motifs = best_motif_in_cluster(
                pwmfile,
                os.path.join(tmpdir, "clustered_motifs.pfm"),
                clusters, 
                os.path.join(tmpdir, 'validation.fa'), 
                background, 
                result.stats)
        
        final_motifs, stats = rename_motifs(best_motifs, result.stats)
    else:
        logger.info("not clustering")
        rank = rank_motifs(result.stats)
        sorted_motifs = sorted(motifs, key=lambda x: rank[str(x)], reverse=True)
        final_motifs, stats = rename_motifs(sorted_motifs, result.stats)

    with open(os.path.join(outdir, "motifs.pwm"), "w") as f:
        for m in final_motifs:
            f.write("{}\n".format(m.to_pwm()))
    
    if create_report:
        bg = dict([(b, os.path.join(tmpdir, "bg.{}.fa".format(b))) for b in background])

        create_denovo_motif_report(
                inputfile, 
                os.path.join(outdir, "motifs.pwm"), 
                os.path.join(tmpdir, "validation.fa"), 
                bg, 
                os.path.join(tmpdir, "localization.fa"), 
                outdir,
                params,
                stats,
                )
    
    with open(os.path.join(outdir, "params.txt"), "w") as f:
        for k,v in params.items():
            f.write("{}\t{}\n".format(k,v))
    
    if not(params.get("keep_intermediate")):
        logger.debug(
            "Deleting intermediate files. "
            "Please specifify the -k option if you want to keep these files.")
        shutil.rmtree(tmpdir)

    logger.info("finished")
    logger.info("output dir: %s", outdir) 
    if cluster:
        logger.info("report: %s", os.path.join(outdir, "motif_report.html"))

    return final_motifs