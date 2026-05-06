def moap(inputfile, method="hypergeom", scoring=None, outfile=None, motiffile=None, pwmfile=None, genome=None, fpr=0.01, ncpus=None,
        subsample=None):
    """Run a single motif activity prediction algorithm.
    
    Parameters
    ----------
    inputfile : str
        :1File with regions (chr:start-end) in first column and either cluster 
        name in second column or a table with values.
    
    method : str, optional
        Motif activity method to use. Any of 'hypergeom', 'lasso', 
        'lightningclassification', 'lightningregressor', 'bayesianridge', 
        'rf', 'xgboost'. Default is 'hypergeom'. 
    
    scoring:  str, optional
        Either 'score' or 'count'
    
    outfile : str, optional
        Name of outputfile to save the fitted activity values.
    
    motiffile : str, optional
        Table with motif scan results. First column should be exactly the same
        regions as in the inputfile.
    
    pwmfile : str, optional
        File with motifs in pwm format. Required when motiffile is not 
        supplied.
    
    genome : str, optional
        Genome name, as indexed by gimme. Required when motiffile is not
        supplied
    
    fpr : float, optional
        FPR for motif scanning
    
    ncpus : int, optional
        Number of threads to use. Default is the number specified in the config.
    
    Returns
    -------
    pandas DataFrame with motif activity
    """

    if scoring and scoring not in ['score', 'count']:
        raise ValueError("valid values are 'score' and 'count'")
    
    config = MotifConfig()

    if inputfile.endswith("feather"):
        df = pd.read_feather(inputfile)
        df = df.set_index(df.columns[0])
    else:
        # read data
        df = pd.read_table(inputfile, index_col=0, comment="#")
    
    clf = Moap.create(method, ncpus=ncpus)

    if clf.ptype == "classification":
        if df.shape[1] != 1:
            raise ValueError("1 column expected for {}".format(method))
    else:
        if np.dtype('object') in set(df.dtypes):
            raise ValueError(
                    "columns should all be numeric for {}".format(method))

    if motiffile is None:
        if genome is None:
            raise ValueError("need a genome")
        
        pwmfile = pwmfile_location(pwmfile)
        try:
            motifs = read_motifs(pwmfile)
        except:
            sys.stderr.write("can't read motifs from {}".format(pwmfile))
            raise

        # initialize scanner
        s = Scanner(ncpus=ncpus)
        sys.stderr.write(pwmfile + "\n")
        s.set_motifs(pwmfile)
        s.set_genome(genome)
        s.set_background(genome=genome)

        # scan for motifs
        sys.stderr.write("scanning for motifs\n")
        motif_names = [m.id for m in read_motifs(pwmfile)]
        scores = []
        if method == 'classic' or scoring == "count":
            s.set_threshold(fpr=fpr)
            for row in s.count(list(df.index)):
                scores.append(row)
        else:
            for row in s.best_score(list(df.index), normalize=True):
                scores.append(row)

        motifs = pd.DataFrame(scores, index=df.index, columns=motif_names)
    else:
        motifs = pd.read_table(motiffile, index_col=0, comment="#")   

    if outfile and os.path.exists(outfile):
        out = pd.read_table(outfile, index_col=0, comment="#")
        ncols = df.shape[1]
        if ncols == 1:
            ncols = len(df.iloc[:,0].unique())
        
        if out.shape[0] == motifs.shape[1] and out.shape[1] == ncols:
            logger.warn("%s output already exists... skipping", method)
            return out
    
    if subsample is not None:
        n = int(subsample * df.shape[0])
        logger.debug("Subsampling %d regions", n)
        df = df.sample(n)
    
    motifs = motifs.loc[df.index]
    
    if method == "lightningregressor":
        outdir = os.path.dirname(outfile)
        tmpname = os.path.join(outdir, ".lightning.tmp")
        clf.fit(motifs, df, tmpdir=tmpname)
        shutil.rmtree(tmpname)
    else:
        clf.fit(motifs, df)
    
    if outfile:
        with open(outfile, "w") as f:
            f.write("# maelstrom - GimmeMotifs version {}\n".format(__version__))
            f.write("# method: {} with motif {}\n".format(method, scoring))
            if genome:
                f.write("# genome: {}\n".format(genome))
            if motiffile:
                f.write("# motif table: {}\n".format(motiffile))
            f.write("# {}\n".format(clf.act_description))
        
        with open(outfile, "a") as f:
            clf.act_.to_csv(f, sep="\t")

    return clf.act_