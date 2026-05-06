def errorprint():
    """Print out descriptions from ConfigurationError."""
    try:
        yield

    except ConfigurationError as e:
        click.secho('%s' % e, err=True, fg='red')
        sys.exit(1)