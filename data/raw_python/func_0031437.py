def parseFASTACommandLineOptions(args):
    """
    Examine parsed command-line options and return a Reads instance.

    @param args: An argparse namespace, as returned by the argparse
        C{parse_args} function.
    @return: A C{Reads} subclass instance, depending on the type of FASTA file
        given.
    """
    # Set default FASTA type.
    if not (args.fasta or args.fastq or args.fasta_ss):
        args.fasta = True

    readClass = readClassNameToClass[args.readClass]

    if args.fasta:
        from dark.fasta import FastaReads
        return FastaReads(args.fastaFile, readClass=readClass)
    elif args.fastq:
        from dark.fastq import FastqReads
        return FastqReads(args.fastaFile, readClass=readClass)
    else:
        from dark.fasta_ss import SSFastaReads
        return SSFastaReads(args.fastaFile, readClass=readClass)