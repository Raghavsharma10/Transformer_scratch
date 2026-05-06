def parse_cutoff(motifs, cutoff, default=0.9):
    """ Provide either a file with one cutoff per motif or a single cutoff
        returns a hash with motif id as key and cutoff as value
    """
    
    cutoffs = {}
    if os.path.isfile(str(cutoff)):
        for i,line in enumerate(open(cutoff)):
            if line != "Motif\tScore\tCutoff\n":
                try:
                    motif,_,c = line.strip().split("\t")
                    c = float(c)
                    cutoffs[motif] = c
                except Exception as e:
                    sys.stderr.write("Error parsing cutoff file, line {0}: {1}\n".format(e, i + 1))
                    sys.exit(1)
    else:
        for motif in motifs:
            cutoffs[motif.id] = float(cutoff)
    
    for motif in motifs:
        if not motif.id in cutoffs:
            sys.stderr.write("No cutoff found for {0}, using default {1}\n".format(motif.id, default))
            cutoffs[motif.id] = default
    return cutoffs