def main(argv=None):
    """The entry point of the application."""
    if argv is None:
        argv = sys.argv[1:]
    usage = '\n\n\n'.join(__doc__.split('\n\n\n')[1:])
    version = 'Nosey ' + __version__

    # Parse options
    args = docopt(usage, argv=argv, version=version)

    # Execute
    return watch(args['<directory>'], args['--clear'])