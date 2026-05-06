def downloaded_reqs_from_path(path, argv):
    """Return a list of DownloadedReqs representing the requirements parsed
    out of a given requirements file.

    :arg path: The path to the requirements file
    :arg argv: The commandline args, starting after the subcommand

    """
    finder = package_finder(argv)
    return [DownloadedReq(req, argv, finder) for req in
            _parse_requirements(path, finder)]