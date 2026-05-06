def parseFASTAEditingCommandLineOptions(args, reads):
    """
    Examine parsed FASTA editing command-line options and return information
    about kept sites and sequences.

    @param args: An argparse namespace, as returned by the argparse
        C{parse_args} function.
    @param reads: A C{Reads} instance to filter.
    @return: The filtered C{Reads} instance.
    """
    removeGaps = args.removeGaps
    removeDescriptions = args.removeDescriptions
    truncateTitlesAfter = args.truncateTitlesAfter
    keepSites = (
        parseRangeString(args.keepSites, convertToZeroBased=True)
        if args.keepSites else None)

    if args.keepSitesFile:
        keepSites = keepSites or set()
        with open(args.keepSitesFile) as fp:
            for lineNumber, line in enumerate(fp):
                try:
                    keepSites.update(
                        parseRangeString(line, convertToZeroBased=True))
                except ValueError as e:
                    raise ValueError(
                        'Keep sites file %r line %d could not be parsed: '
                        '%s' % (args.keepSitesFile, lineNumber, e))

    removeSites = (
        parseRangeString(args.removeSites, convertToZeroBased=True)
        if args.removeSites else None)

    if args.removeSitesFile:
        removeSites = removeSites or set()
        with open(args.removeSitesFile) as fp:
            for lineNumber, line in enumerate(fp):
                try:
                    removeSites.update(
                        parseRangeString(line, convertToZeroBased=True))
                except ValueError as e:
                    raise ValueError(
                        'Remove sites file %r line %d parse error: %s'
                        % (args.removeSitesFile, lineNumber, e))

    return reads.filter(
        removeGaps=removeGaps,
        truncateTitlesAfter=truncateTitlesAfter,
        removeDescriptions=removeDescriptions,
        idLambda=args.idLambda, readLambda=args.readLambda,
        keepSites=keepSites, removeSites=removeSites,
        reverse=args.reverse, reverseComplement=args.reverseComplement)