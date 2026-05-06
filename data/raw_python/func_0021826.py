def setup_opt_parser():
    """
    Setup the optparser

    @returns: opt_parser.OptionParser

    """
    #pylint: disable-msg=C0301
    #line too long

    usage = "usage: %prog [options]"
    opt_parser = optparse.OptionParser(usage=usage)

    opt_parser.add_option("--version", action='store_true', dest=
                          "yolk_version", default=False, help=
                          "Show yolk version and exit.")

    opt_parser.add_option("--debug", action='store_true', dest=
                          "debug", default=False, help=
                          "Show debugging information.")

    opt_parser.add_option("-q", "--quiet", action='store_true', dest=
                          "quiet", default=False, help=
                          "Show less output.")
    group_local = optparse.OptionGroup(opt_parser,
            "Query installed Python packages",
            "The following options show information about installed Python packages. Activated packages are normal packages on sys.path that can be imported. Non-activated packages need 'pkg_resources.require()' before they can be imported, such as packages installed with 'easy_install --multi-version'. PKG_SPEC can be either a package name or package name and version e.g. Paste==0.9")

    group_local.add_option("-l", "--list", action='store_true', dest=
                           "show_all", default=False, help=
                           "List all Python packages installed by distutils or setuptools. Use PKG_SPEC to narrow results.")

    group_local.add_option("-a", "--activated", action='store_true',
                           dest="show_active", default=False, help=
                           'List activated packages installed by distutils or ' +
                           'setuptools. Use PKG_SPEC to narrow results.')

    group_local.add_option("-n", "--non-activated", action='store_true',
                           dest="show_non_active", default=False, help=
                           'List non-activated packages installed by distutils or ' +
                           'setuptools. Use PKG_SPEC to narrow results.')

    group_local.add_option("-m", "--metadata", action='store_true', dest=
                           "metadata", default=False, help=
                           'Show all metadata for packages installed by ' +
                           'setuptools (use with -l -a or -n)')

    group_local.add_option("-f", "--fields", action="store", dest=
                           "fields", default=False, help=
                           'Show specific metadata fields. ' +
                           '(use with -m or -M)')

    group_local.add_option("-d", "--depends", action='store', dest=
                           "show_deps", metavar='PKG_SPEC',
                           help= "Show dependencies for a package installed by " +
                           "setuptools if they are available.")

    group_local.add_option("--entry-points", action='store',
                           dest="show_entry_points", default=False, help=
                           'List entry points for a module. e.g. --entry-points nose.plugins',
                            metavar="MODULE")

    group_local.add_option("--entry-map", action='store',
                           dest="show_entry_map", default=False, help=
                           'List entry map for a package. e.g. --entry-map yolk',
                           metavar="PACKAGE_NAME")
    group_pypi = optparse.OptionGroup(opt_parser,
            "PyPI (Cheese Shop) options",
            "The following options query the Python Package Index:")

    group_pypi.add_option("-C", "--changelog", action='store',
                          dest="show_pypi_changelog", metavar='HOURS',
                          default=False, help=
                          "Show detailed ChangeLog for PyPI for last n hours. ")

    group_pypi.add_option("-D", "--download-links", action='store',
                          metavar="PKG_SPEC", dest="show_download_links",
                          default=False, help=
                          "Show download URL's for package listed on PyPI. Use with -T to specify egg, source etc.")

    group_pypi.add_option("-F", "--fetch-package", action='store',
                          metavar="PKG_SPEC", dest="fetch",
                          default=False, help=
                          "Download package source or egg. You can specify a file type with -T")

    group_pypi.add_option("-H", "--browse-homepage", action='store',
                          metavar="PKG_SPEC", dest="browse_website",
                          default=False, help=
                          "Launch web browser at home page for package.")

    group_pypi.add_option("-I", "--pypi-index", action='store',
                          dest="pypi_index",
                          default=False, help=
                          "Specify PyPI mirror for package index.")

    group_pypi.add_option("-L", "--latest-releases", action='store',
                          dest="show_pypi_releases", metavar="HOURS",
                          default=False, help=
                          "Show PyPI releases for last n hours. ")

    group_pypi.add_option("-M", "--query-metadata", action='store',
                          dest="query_metadata_pypi", default=False,
                          metavar="PKG_SPEC", help=
                          "Show metadata for a package listed on PyPI. Use -f to show particular fields.")

    group_pypi.add_option("-S", "", action="store", dest="pypi_search",
                          default=False, help=
                          "Search PyPI by spec and optional AND/OR operator.",
                          metavar='SEARCH_SPEC <AND/OR SEARCH_SPEC>')

    group_pypi.add_option("-T", "--file-type", action="store", dest=
                          "file_type", default="all", help=
                          "You may specify 'source', 'egg', 'svn' or 'all' when using -D.")

    group_pypi.add_option("-U", "--show-updates", action='store_true',
                          dest="show_updates", metavar='<PKG_NAME>',
                          default=False, help=
                          "Check PyPI for updates on package(s).")

    group_pypi.add_option("-V", "--versions-available", action=
                          'store', dest="versions_available",
                          default=False, metavar='PKG_SPEC',
                          help="Show available versions for given package " +
                          "listed on PyPI.")
    opt_parser.add_option_group(group_local)
    opt_parser.add_option_group(group_pypi)
    # add opts from plugins
    all_plugins = []
    for plugcls in load_plugins(others=True):
        plug = plugcls()
        try:
            plug.add_options(opt_parser)
        except AttributeError:
            pass

    return opt_parser