def main():
    "Boots up the command line tool"
    logging.captureWarnings(True)
    args = build_parser().parse_args()
    # Configure logging
    args.setup_logging(args)
    # Dispatch into the appropriate subcommand function.
    try:
        return args.func(args)
    except SystemExit:
        raise
    except:
        logging.exception('Problem when running command. Sorry!')
        sys.exit(1)