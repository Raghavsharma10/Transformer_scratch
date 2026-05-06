def main(verbose, **kwargs):
    """Replace strings in one or multiple files.

    You must either provide `REGEX_PATH` or use the `-c` flag
    to provide a valid repex configuration.

    `REGEX_PATH` can be: a regex of paths under `basedir`,
    a path to a single directory under `basedir`,
    or a path to a single file.

    It's important to note that if the `REGEX_PATH` is a path to a
    directory, the `-t,--ftype` flag must be provided.
    """
    config = kwargs['config']

    if not config and not kwargs['regex_path']:
        click.echo('Must either provide a path or a viable repex config file.')
        sys.exit(1)

    if verbose:
        set_verbose()

    if config:
        repex_vars = _build_vars_dict(kwargs['vars_file'], kwargs['var'])
        try:
            iterate(
                config_file_path=config,
                variables=repex_vars,
                tags=list(kwargs['tag']),
                validate=kwargs['validate'],
                validate_only=kwargs['validate_only'],
                with_diff=kwargs['diff'])
        except (RepexError, IOError, OSError) as ex:
            sys.exit(str(ex))
    else:
        pathobj = _construct_path_object(**kwargs)
        try:
            handle_path(pathobj)
        except (RepexError, IOError, OSError) as ex:
            sys.exit(str(ex))