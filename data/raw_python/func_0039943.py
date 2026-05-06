def main(argv=None):
    """
    Entry point

    :param argv: Program arguments
    """
    parser = argparse.ArgumentParser(description="Multicast packets spy")

    parser.add_argument("-g", "--group", dest="group", default="239.0.0.1",
                        help="Multicast target group (address)")
    parser.add_argument("-p", "--port", type=int, dest="port", default=42000,
                        help="Multicast target port")

    parser.add_argument("-d", "--debug", action="store_true", dest="debug",
                        help="Set logger to DEBUG level")
    parser.add_argument("-v", "--verbose", action="store_true", dest="verbose",
                        help="Verbose output")

    # Parse arguments
    args = parser.parse_args(argv)

    # Configure the logger
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.WARNING)

    try:
        return run_spy(args.group, args.port, args.verbose)
    except Exception as ex:
        logging.exception("Error running spy: %s", ex)

    return 1