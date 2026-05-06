def parseFASTAFilteringCommandLineOptions(args, reads):
    """
    Examine parsed FASTA filtering command-line options and return filtered
    reads.

    @param args: An argparse namespace, as returned by the argparse
        C{parse_args} function.
    @param reads: A C{Reads} instance to filter.
    @return: The filtered C{Reads} instance.
    """
    keepSequences = (
        parseRangeString(args.keepSequences, convertToZeroBased=True)
        if args.keepSequences else None)

    removeSequences = (
        parseRangeString(args.removeSequences, convertToZeroBased=True)
        if args.removeSequences else None)

    return reads.filter(
        minLength=args.minLength, maxLength=args.maxLength,
        whitelist=set(args.whitelist) if args.whitelist else None,
        blacklist=set(args.blacklist) if args.blacklist else None,
        whitelistFile=args.whitelistFile, blacklistFile=args.blacklistFile,
        titleRegex=args.titleRegex,
        negativeTitleRegex=args.negativeTitleRegex,
        keepSequences=keepSequences, removeSequences=removeSequences,
        head=args.head, removeDuplicates=args.removeDuplicates,
        removeDuplicatesById=args.removeDuplicatesById,
        randomSubset=args.randomSubset, trueLength=args.trueLength,
        sampleFraction=args.sampleFraction,
        sequenceNumbersFile=args.sequenceNumbersFile)