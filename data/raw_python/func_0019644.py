def scan_to_table(input_table, genome, scoring, pwmfile=None, ncpus=None):
    """Scan regions in input table with motifs.

    Parameters
    ----------
    input_table : str
        Filename of input table. Can be either a text-separated tab file or a
        feather file.
    
    genome : str
        Genome name. Can be either the name of a FASTA-formatted file or a 
        genomepy genome name.
    
    scoring : str
        "count" or "score"
    
    pwmfile : str, optional
        Specify a PFM file for scanning.
    
    ncpus : int, optional
        If defined this specifies the number of cores to use.
    
    Returns
    -------
    table : pandas.DataFrame
        DataFrame with motif ids as column names and regions as index. Values
        are either counts or scores depending on the 'scoring' parameter.s
    """
    config = MotifConfig()
    
    if pwmfile is None:
        pwmfile = config.get_default_params().get("motif_db", None)
        if pwmfile is not None:
            pwmfile = os.path.join(config.get_motif_dir(), pwmfile)

    if pwmfile is None:
        raise ValueError("no pwmfile given and no default database specified")

    logger.info("reading table")
    if input_table.endswith("feather"):
        df = pd.read_feather(input_table)
        idx = df.iloc[:,0].values
    else:
        df = pd.read_table(input_table, index_col=0, comment="#")
        idx = df.index
    
    regions = list(idx)
    s = Scanner(ncpus=ncpus)
    s.set_motifs(pwmfile)
    s.set_genome(genome)
    s.set_background(genome=genome)
    
    nregions = len(regions)

    scores = []
    if scoring == "count":
        logger.info("setting threshold")
        s.set_threshold(fpr=FPR)
        logger.info("creating count table")
        for row in s.count(regions):
            scores.append(row)
        logger.info("done")
    else:
        s.set_threshold(threshold=0.0)
        logger.info("creating score table")
        for row in s.best_score(regions, normalize=True):
            scores.append(row)
        logger.info("done")
   
    motif_names = [m.id for m in read_motifs(pwmfile)]
    logger.info("creating dataframe")
    return pd.DataFrame(scores, index=idx, columns=motif_names)