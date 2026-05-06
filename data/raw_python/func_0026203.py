def _processCommandLineArgs():
    """
        Get the command line arguments

        Parameters: NONE

        Returns:

           files            list of file specifications to be converted

           outputFileNames  list of output file specifications
                             (one per input file)
                             Default: a list of None values (one per input file)

           conversionFormat string indicating the conversion format requested
                             Default: "mulitextension"

           verbose          flag indicating if verbose output is desired
                             Default: False

        Exceptions: NONE
    """

    import getopt

    try:
        opts, args = getopt.getopt(sys.argv[1:], "hvmo:",
                                   ["help",
                                    "verbose",
                                    "multiExtensionConversion",
                                    "outputFileName"])
    except getopt.GetoptError as e:
        print(str(e))
        _usage()
        sys.exit(1)

    conversionFormat = ""
    outputFileNames = []
    verbose = False

    for o, a in opts:
        if o in ("-h", "--help"):
            _usage()
            print("       Convert the waivered FITS Files (FILEs) to various formats.")
            print("       The default conversion format is multi-extension FITS.")
            print("       Options:")
            print("         -h,  --help                       display this help message and exit")
            print("         -v,  --verbose                    provide verbose output")
            print("         -m,  --multiExtensionConversion   convert to multiExtension FITS format")
            print("         -o,  --outputFileName             comma separated list of output file")
            print("                                           specifications (one per input FILE)")
            sys.exit()

        if o in ("-v", "--verbose"):
            verbose = True

        if o in ("-m", "--multiExtensionConversion"):
            if conversionFormat != "":
                print("convertwaiveredfits.py: only one conversion format allowed")
                _usage()
                sys.exit(1)

            conversionFormat = "multiExtension"

        if o in ("-o", "--outputFileName"):
            outputFileNames = a.split(',')

    if conversionFormat == "":
        #
        # Set the default conversion format if none was provided
        #
        conversionFormat = "multiExtension"

    if not args:
        print("convertwaiveredfits.py: nothing to convert")
        _usage()
        sys.exit(1)
    else:
        files = args

        if outputFileNames:
            if len(files) != len(outputFileNames):
                print("convertwaiveredfits.py: number of output file names does not match")
                print("                        the number of FILEs to convert")
                _usage()
                sys.exit(1)
        else:
            for i in range(0,len(files)):
                outputFileNames.append(None)

    return files,outputFileNames,conversionFormat,verbose