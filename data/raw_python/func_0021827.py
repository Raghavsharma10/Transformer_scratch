def validate_pypi_opts(opt_parser):
    """
    Check parse options that require pkg_spec

    @returns: pkg_spec

    """

    (options, remaining_args) = opt_parser.parse_args()
    options_pkg_specs = [ options.versions_available,
            options.query_metadata_pypi,
            options.show_download_links,
            options.browse_website,
            options.fetch,
            options.show_deps,
            ]
    for pkg_spec in options_pkg_specs:
        if pkg_spec:
            return pkg_spec