def maelstrom(args):
    """Run the maelstrom method."""
    infile = args.inputfile
    genome = args.genome
    outdir = args.outdir
    pwmfile = args.pwmfile
    methods = args.methods
    ncpus = args.ncpus
    
    if not os.path.exists(infile):
        raise ValueError("file {} does not exist".format(infile))

    if methods:
        methods = [x.strip() for x in methods.split(",")]

    run_maelstrom(infile, genome, outdir, pwmfile, methods=methods, ncpus=ncpus)