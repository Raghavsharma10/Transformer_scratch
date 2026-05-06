def location(args):
    """
    Creates histrogram of motif location.

    Parameters
    ----------
    args : argparse object
        Command line arguments.
    """
    fastafile = args.fastafile
    pwmfile = args.pwmfile

    lwidth = args.width
    if not lwidth:
        f = Fasta(fastafile)
        lwidth = len(f.items()[0][1])
        f = None

    jobs = []
    motifs = pwmfile_to_motifs(pwmfile)
    ids = [motif.id for motif in motifs]
    if args.ids:
        ids = args.ids.split(",")
    
    n_cpus = int(MotifConfig().get_default_params()["ncpus"])
    pool = Pool(processes=n_cpus, maxtasksperchild=1000) 
    for motif in motifs:
        if motif.id in ids:
            outfile = os.path.join("%s_histogram" % motif.id)
            jobs.append(
                    pool.apply_async(
                        motif_localization, 
                        (fastafile,motif,lwidth,outfile, args.cutoff)
                        ))
    
    for job in jobs:
        job.get()