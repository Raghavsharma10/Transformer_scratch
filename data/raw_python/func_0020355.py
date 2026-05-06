def main(argv):
    """Main"""
    global options

    opts = None
    try:
        opts, args = getopt.getopt(argv, options['short'], options['long'])
    except getopt.GetoptError:
        usage()
        exit(2)

    for opt, arg in opts:
        if opt in ("-h", "--help"):
            usage()
            exit()
        elif opt in ("-d", "--debug"):
            try:
                arg = int(arg)
                log.debug("Debug level received: " + str(arg))
            except ValueError:
                log.warning("Invalid log level: " + arg)
                continue

            if 0 <= arg <= 5:
                log.setLevel(60 - (arg*10))
                log.critical("Log level changed to: " + str(logging.getLevelName(60 - (arg*10))))
            else:
                log.warning("Invalid log level: " + str(arg))

    infile = None
    outfile = None
    remove_background = False
    duration_format = False
    deduplicate = False

    for opt, arg in opts:
        if opt in ("-i", "--infile"):
            log.info("Input File: " + arg)
            infile = arg
        if opt in ("-o", "--outfile"):
            log.info("Output File: " + arg)
            outfile = arg
        if opt in ("-r", "--remove-background"):
            log.info("Remove Background: Enabled")
            remove_background = True
        if opt in ("-f", "--format-duration"):
            log.info("Format Duration: Enabled")
            duration_format = True
        if opt in ("-D", "--deduplicate"):
            log.info("Deduplicate: Enabled")
            deduplicate = True

    if infile is None:
        log.critical("No input JSON provided.")
        usage()
        exit(3)

    with open(infile) as f:
        cucumber_output = convert(json.load(f),
                                  remove_background=remove_background,
                                  duration_format=duration_format,
                                  deduplicate=deduplicate)

    if outfile is not None:
        with open(outfile, 'w') as f:
            json.dump(cucumber_output, f, indent=4, separators=(',', ': '))
    else:
        pprint(cucumber_output)