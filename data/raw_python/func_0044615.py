def main():
    """Collect user args and call command funct.

    Collect command line args and setup environment then call
    function for command specified in args.

    """
    parser = parser_setup()
    options = parser.parse_args()

    debug = bool(options.debug > 0)
    debugall = bool(options.debug > 1)

    awsc.init()
    debg.init(debug, debugall)
    print(C_NORM)

    options.func(options)

    sys.exit()