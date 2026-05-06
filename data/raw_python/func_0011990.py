def check_options(options, parser):
    """
    check options requirements, print and return exit value
    """
    if not options.get('release_environment', None):
        print("release environment is required")
        parser.print_help()
        return os.EX_USAGE

    return 0