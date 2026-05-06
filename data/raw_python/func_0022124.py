def main(argv=None):  # pragma: no coverage
    """ Main entry point when the user runs the `trytravis` command. """
    try:
        colorama.init()
        if argv is None:
            argv = sys.argv[1:]
        _main(argv)
    except RuntimeError as e:
        print(colorama.Fore.RED + 'ERROR: ' +
              str(e) + colorama.Style.RESET_ALL)
        sys.exit(1)
    else:
        sys.exit(0)