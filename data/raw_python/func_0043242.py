def ls(args):
    """
    List sites
    ----------

    Show list of installed sites.

    ::

        usage: makesite ls [-h] [-v] [-p PATH]

        Show list of installed sites.

        optional arguments:
        -p PATH, --path PATH  path to makesite sites instalation dir. you can set it
                                in $makesite_home env variable.

    Examples: ::

            makesite ls

    """

    assert args.path, "Not finded MAKESITE HOME."

    print_header("Installed sites:")
    for site in gen_sites(args.path):
        LOGGER.debug(site.get_info())
    return True