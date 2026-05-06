def delim(arguments):
    """
    Execute delim action.

    :param arguments: Parsed command line arguments from :func:`main`
    """

    if bool(arguments.control_files) == bool(arguments.directory):
        raise ValueError(
                'Exactly one of control_files and `-d` must be specified.')

    if arguments.directory:
        arguments.control_files.extend(control_iter(arguments.directory))

    with arguments.output as fp:
        results = _delim_accum(arguments.control_files,
                arguments.file_template, arguments.keys,
                arguments.exclude_keys, arguments.separator,
                missing_action=arguments.missing_action)
        r = next(results)
        writer = csv.DictWriter(fp, r.keys(), delimiter=arguments.separator)
        writer.writeheader()
        writer.writerow(r)
        writer.writerows(results)