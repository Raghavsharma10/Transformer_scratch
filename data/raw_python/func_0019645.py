def run_maelstrom(infile, genome, outdir, pwmfile=None, plot=True, cluster=False, 
        score_table=None, count_table=None, methods=None, ncpus=None):
    """Run maelstrom on an input table.
    
    Parameters
    ----------
    infile : str
        Filename of input table. Can be either a text-separated tab file or a
        feather file.
    
    genome : str
        Genome name. Can be either the name of a FASTA-formatted file or a 
        genomepy genome name.
    
    outdir : str
        Output directory for all results.

    pwmfile : str, optional
        Specify a PFM file for scanning.

    plot : bool, optional
        Create heatmaps.
    
    cluster : bool, optional
        If True and if the input table has more than one column, the data is
        clustered and the cluster activity methods are also run. Not 
        well-tested.
    
    score_table : str, optional
        Filename of pre-calculated table with motif scores.

    count_table : str, optional
        Filename of pre-calculated table with motif counts.

    methods : list, optional
        Activity methods to use. By default are all used.

    ncpus : int, optional
        If defined this specifies the number of cores to use.
    """
    logger.info("Starting maelstrom")
    if infile.endswith("feather"):
        df = pd.read_feather(infile)
        df = df.set_index(df.columns[0])
    else:
        df = pd.read_table(infile, index_col=0, comment="#")
    
    # Check for duplicates
    if df.index.duplicated(keep=False).any():
        raise ValueError("Input file contains duplicate regions! "
                         "Please remove them.")
    
    if not os.path.exists(outdir):
        os.mkdir(outdir)

    if methods is None:
        methods = Moap.list_predictors() 
    methods = [m.lower() for m in methods]

    shutil.copyfile(infile, os.path.join(outdir, "input.table.txt"))
    
    # Copy the motif informatuon
    pwmfile = pwmfile_location(pwmfile) 
    if pwmfile:
        shutil.copy2(pwmfile, outdir)
        mapfile = re.sub(".p[fw]m$", ".motif2factors.txt", pwmfile)
        if os.path.exists(mapfile):
            shutil.copy2(mapfile, outdir)
    
    # Create a file with the number of motif matches
    if not count_table:
        count_table = os.path.join(outdir, "motif.count.txt.gz")
        if not os.path.exists(count_table):
            logger.info("Motif scanning (counts)")
            counts = scan_to_table(infile, genome, "count",
                pwmfile=pwmfile, ncpus=ncpus)
            counts.to_csv(count_table, sep="\t", compression="gzip")
        else:
            logger.info("Counts, using: %s", count_table)

    # Create a file with the score of the best motif match
    if not score_table:
        score_table = os.path.join(outdir, "motif.score.txt.gz")
        if not os.path.exists(score_table):
            logger.info("Motif scanning (scores)")
            scores = scan_to_table(infile, genome, "score",
                pwmfile=pwmfile, ncpus=ncpus)
            scores.to_csv(score_table, sep="\t", float_format="%.3f", 
                compression="gzip")
        else:
            logger.info("Scores, using: %s", score_table)

    if cluster:
        cluster = False
        for method in methods:
            m = Moap.create(method, ncpus=ncpus)
            if m.ptype == "classification":
                cluster = True
                break
        if not cluster:
            logger.info("Skipping clustering, no classification methods")
    
    exps = []
    clusterfile = infile
    if df.shape[1] != 1:
        # More than one column
        for method in Moap.list_regression_predictors():
            if method in methods:
                m = Moap.create(method, ncpus=ncpus)
                exps.append([method, m.pref_table, infile])
                logger.debug("Adding %s", method)

        if cluster:
            clusterfile = os.path.join(outdir,
                    os.path.basename(infile) + ".cluster.txt")
            
            df[:] = scale(df, axis=0)
            names = df.columns
            df_changed = pd.DataFrame(index=df.index)
            df_changed["cluster"] = np.nan
            for name in names:
                df_changed.loc[(df[name] - df.loc[:,df.columns != name].max(1)) > 0.5, "cluster"] = name
            df_changed.dropna().to_csv(clusterfile, sep="\t")
    if df.shape[1] == 1 or cluster:
        for method in Moap.list_classification_predictors():
            if method in methods:
                m = Moap.create(method, ncpus=ncpus)
                exps.append([method, m.pref_table, clusterfile])

    if len(exps) == 0:
        logger.error("No method to run.")
        sys.exit(1)

    for method, scoring, fname in exps:
        try:
            if scoring == "count" and count_table:
                moap_with_table(fname, count_table, outdir, method, scoring, ncpus=ncpus)
            elif scoring == "score" and score_table:
                moap_with_table(fname, score_table, outdir, method, scoring, ncpus=ncpus)
            else:
                moap_with_bg(fname, genome, outdir, method, scoring, pwmfile=pwmfile, ncpus=ncpus)
        
        except Exception as e:
            logger.warn("Method %s with scoring %s failed", method, scoring)
            logger.warn(e)
            logger.warn("Skipping")
            raise 
    dfs = {}
    for method, scoring,fname  in exps:
        t = "{}.{}".format(method,scoring)
        fname = os.path.join(outdir, "activity.{}.{}.out.txt".format(
                           method, scoring))
        try:
            dfs[t] = pd.read_table(fname, index_col=0, comment="#")
        except:
            logging.warn("Activity file for {} not found!\n".format(t))
   
    if len(methods) > 1:
        logger.info("Rank aggregation")
        df_p = df_rank_aggregation(df, dfs, exps)
        df_p.to_csv(os.path.join(outdir, "final.out.csv"), sep="\t")
    #df_p = df_p.join(m2f)

    # Write motif frequency table
    
    if df.shape[1] == 1:
        mcount = df.join(pd.read_table(count_table, index_col=0, comment="#"))
        m_group = mcount.groupby(df.columns[0])
        freq = m_group.sum() / m_group.count()
        freq.to_csv(os.path.join(outdir, "motif.freq.txt"), sep="\t")

    if plot and len(methods) > 1:
        logger.info("html report")
        maelstrom_html_report(
                outdir, 
                os.path.join(outdir, "final.out.csv"),
                pwmfile
                )
        logger.info(os.path.join(outdir, "gimme.maelstrom.report.html"))