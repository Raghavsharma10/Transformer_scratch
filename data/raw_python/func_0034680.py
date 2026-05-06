def main():
    """Parser the command line and run the validator."""
    parser = argparse.ArgumentParser(
        description="[v" + __version__ + "] " + __doc__,
        prog="w3c_validator",
    )
    parser.add_argument(
        "--log",
        default="INFO",
        help=("log level: DEBUG, INFO or INFO "
              "(default: INFO)"))
    parser.add_argument(
        "--version", action="version", version="%(prog)s " + __version__)
    parser.add_argument(
        "--verbose", help="increase output verbosity", action="store_true")
    parser.add_argument(
        "source", metavar="F", type=str, nargs="+", help="file or URL")
    args = parser.parse_args()

    logging.basicConfig(level=getattr(logging, args.log))

    LOGGER.info("Files to validate: \n  {0}".format("\n  ".join(args.source)))
    LOGGER.info("Number of files: {0}".format(len(args.source)))

    errors = 0
    warnings = 0
    for f in args.source:
        LOGGER.info("validating: %s ..." % f)
        retrys = 0
        while retrys < 2:
            result = validate(f, verbose=args.verbose)
            if result:
                break

            time.sleep(2)
            retrys += 1
            LOGGER.info("retrying: %s ..." % f)
        else:
            LOGGER.info("failed: %s" % f)
            errors += 1
            continue

        # import pdb; pdb.set_trace()
        if f.endswith(".css"):
            errorcount = result["cssvalidation"]["result"]["errorcount"]
            warningcount = result["cssvalidation"]["result"]["warningcount"]
            errors += errorcount
            warnings += warningcount
            if errorcount > 0:
                LOGGER.info("errors: %d" % errorcount)
            if warningcount > 0:
                LOGGER.info("warnings: %d" % warningcount)
        else:
            for msg in result["messages"]:
                print_msg(msg)
                if msg["type"] == "error":
                    errors += 1
                else:
                    warnings += 1
    sys.exit(min(errors, 255))