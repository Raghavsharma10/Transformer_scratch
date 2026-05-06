def main():
    """
    Entry point.
    """
    parser = argparse.ArgumentParser(description=DESCRIPTION)
    for arg in ARGUMENTS:
        if "action" in arg:
            if arg["short"] is not None:
                parser.add_argument(arg["short"], arg["long"], action=arg["action"], help=arg["help"])
            else:
                parser.add_argument(arg["long"], action=arg["action"], help=arg["help"])
        else:
            if arg["short"] is not None:
                parser.add_argument(arg["short"], arg["long"], nargs=arg["nargs"], type=arg["type"], default=arg["default"], help=arg["help"])
            else:
                parser.add_argument(arg["long"], nargs=arg["nargs"], type=arg["type"], default=arg["default"], help=arg["help"])
    vargs = vars(parser.parse_args())
    command = vargs["command"]
    string = to_unicode_string(vargs["string"])
    if command not in COMMAND_MAP:
        parser.print_help()
        sys.exit(2)
    COMMAND_MAP[command](string, vargs)
    sys.exit(0)