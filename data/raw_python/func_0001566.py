def main():
    """
    Entry point for the standalone script.

    """
    (options, strings) = parseArgv()
    global _suffixArray, _trace

    #############
    # Verbosity #
    #############
    _trace = options.verbose

    ###################
    # Processing unit #
    ###################
    if options.unit == "byte":
        options.unit = UNIT_BYTE
    elif options.unit == "character":
        options.unit = UNIT_CHARACTER
    elif options.unit == "word":
        options.unit = UNIT_WORD
    else:
        print >> _stderr, "Please specify a valid unit type."
        exit(EXIT_BAD_OPTION)

    ######################
    # Build suffix array #
    ######################
    if not options.SAFile:  # Build the suffix array from INPUT
        if not options.input:  # default is standard input
            options.input = "-"
        try:
            string = _open(options.input, "r").read()
        except IOError:
            print >> _stderr, "File %s does not exist." % options.input
            exit(EXIT_ERROR_FILE)

        SA = SuffixArray(string, options.unit, options.encoding, options.noLCPs)
    ########################
    # Or load suffix array #
    ########################
    elif not options.input and options.SAFile:  # Load suffix array from SA_FILE
        try:
            SA = SuffixArray.fromFile(options.SAFile)
        except IOError:
            print >> _stderr, "SA_FILE %s does not exist." % options.SAFile
            exit(EXIT_ERROR_FILE)
    else:
        print >> _stderr, "Please set only one option amongst --input and --load.\n" + \
        "Type %s --help for more details." % _argv[0]
        exit(EXIT_BAD_OPTION)

    ######################
    # Print suffix array #
    ######################
    if options.printSA:
        # Buffered ouptut
        deltaLength = 1000
        start = 0
        while start < SA.length:
            print >> _stderr, SA.__str__(start, start + deltaLength)
            start += deltaLength

    ####################################
    # Look for every string in strings #
    ####################################
    for string in strings:
        print >> _stderr, ""
        print >> _stderr, "Positions of %s:" % string
        print >> _stderr, "  %s" % list(SA.find(string))

    #########################
    # Save SAFILE if needed #
    #########################
    if options.output:
        SA.toFile(options.output)

    if _trace: print >> _stderr, "Done\r\n"