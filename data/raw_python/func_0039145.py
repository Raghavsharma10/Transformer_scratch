def _build_package_finder(options, index_urls, session):
        """
        Create a package finder appropriate to this list command.
        """
        return PackageFinder(
            find_links=options.get('find_links'),
            index_urls=index_urls,
            allow_all_prereleases=options.get('pre'),
            trusted_hosts=options.get('trusted_hosts'),
            session=session,
        )