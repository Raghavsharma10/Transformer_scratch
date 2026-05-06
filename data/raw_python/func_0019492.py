def create_background(bg_type, fafile, outfile, genome="hg18", width=200, nr_times=10, custom_background=None):
    """Create background of a specific type.

    Parameters
    ----------
    bg_type : str
        Name of background type.

    fafile : str
        Name of input FASTA file.

    outfile : str
        Name of output FASTA file.

    genome : str, optional
        Genome name.

    width : int, optional
        Size of regions.

    nr_times : int, optional
        Generate this times as many background sequences as compared to 
        input file.
    
    Returns
    -------
    nr_seqs  : int
        Number of sequences created.
    """
    width = int(width)
    config = MotifConfig()
    fg = Fasta(fafile)

    if bg_type in ["genomic", "gc"]:
        if not genome:
            logger.error("Need a genome to create background")
            sys.exit(1)
    
    if bg_type == "random":
        f = MarkovFasta(fg, k=1, n=nr_times * len(fg))
        logger.debug("Random background: %s", outfile)
    elif bg_type == "genomic":
        logger.debug("Creating genomic background")
        f = RandomGenomicFasta(genome, width, nr_times * len(fg))
    elif bg_type == "gc":
        logger.debug("Creating GC matched background")
        f = MatchedGcFasta(fafile, genome, nr_times * len(fg))
        logger.debug("GC matched background: %s", outfile)
    elif bg_type == "promoter":
        fname = Genome(genome).filename
        gene_file = fname.replace(".fa", ".annotation.bed.gz")
        if not gene_file:
            gene_file = os.path.join(config.get_gene_dir(), "%s.bed" % genome)
        if not os.path.exists(gene_file):
            print("Could not find a gene file for genome {}")
            print("Did you use the --annotation flag for genomepy?")
            print("Alternatively make sure there is a file called {}.bed in {}".format(genome, config.get_gene_dir()))
            raise ValueError()

        logger.info(
                "Creating random promoter background (%s, using genes in %s)",
                genome, gene_file)
        f = PromoterFasta(gene_file, genome, width, nr_times * len(fg))
        logger.debug("Random promoter background: %s", outfile)
    elif bg_type == "custom":
        bg_file = custom_background
        if not bg_file:
            raise IOError(
                    "Background file not specified!")

        if not os.path.exists(bg_file):
            raise IOError(
                    "Custom background file %s does not exist!",
                    bg_file)
        else:
            logger.info("Copying custom background file %s to %s.",
                    bg_file, outfile)
            f = Fasta(bg_file)
            l = np.median([len(seq) for seq in f.seqs])
            if l < (width * 0.95) or l > (width * 1.05):
                   logger.warn(
                    "The custom background file %s contains sequences with a "
                    "median length of %s, while GimmeMotifs predicts motifs in sequences "
                    "of length %s. This will influence the statistics! It is recommended "
                    "to use background sequences of the same length.", 
                    bg_file, l, width)
    
    f.writefasta(outfile)
    return len(f)