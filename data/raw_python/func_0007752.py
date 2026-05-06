def cli(input, verbose, quiet, output_format, precision, indent):
    """Convert text read from the first positional argument, stdin, or
    a file to GeoJSON and write to stdout."""

    verbosity = verbose - quiet
    configure_logging(verbosity)
    logger = logging.getLogger('geomet')

    # Handle the case of file, stream, or string input.
    try:
        src = click.open_file(input).readlines()
    except IOError:
        src = [input]

    stdout = click.get_text_stream('stdout')

    # Read-write loop.
    try:
        for line in src:
            text = line.strip()
            logger.debug("Input: %r", text)
            output = translate(
                text,
                output_format=output_format,
                indent=indent,
                precision=precision
            )
            logger.debug("Output: %r", output)
            stdout.write(output)
            stdout.write('\n')
        sys.exit(0)
    except Exception:
        logger.exception("Failed. Exception caught")
        sys.exit(1)