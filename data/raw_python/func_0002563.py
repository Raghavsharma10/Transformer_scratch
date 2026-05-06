def build_suite(args):
    """Build a test suite by loading TAP files or a TAP stream."""
    loader = Loader()
    if len(args.files) == 0 or args.files[0] == "-":
        suite = loader.load_suite_from_stdin()
    else:
        suite = loader.load(args.files)
    return suite