def run(self):
        """
        Perform actions based on CLI options

        @returns: status code
        """
        opt_parser = setup_opt_parser()
        (self.options, remaining_args) = opt_parser.parse_args()
        logger = self.set_log_level()

        pkg_spec = validate_pypi_opts(opt_parser)
        if not pkg_spec:
            pkg_spec = remaining_args
        self.pkg_spec = pkg_spec

        if not self.options.pypi_search and (len(sys.argv) == 1 or\
                len(remaining_args) > 2):
            opt_parser.print_help()
            return 2

        #Options that depend on querying installed packages, not PyPI.
        #We find the proper case for package names if they are installed,
        #otherwise PyPI returns the correct case.
        if self.options.show_deps or self.options.show_all or \
                self.options.show_active or self.options.show_non_active  or \
                (self.options.show_updates and pkg_spec):
            want_installed = True
        else:
            want_installed = False
        #show_updates may or may not have a pkg_spec
        if not want_installed or self.options.show_updates:
            self.pypi = CheeseShop(self.options.debug)
            #XXX: We should return 2 here if we couldn't create xmlrpc server

        if pkg_spec:
            (self.project_name, self.version, self.all_versions) = \
                    self.parse_pkg_ver(want_installed)
            if want_installed and not self.project_name:
                logger.error("%s is not installed." % pkg_spec[0])
                return 1

        #I could prefix all these with 'cmd_' and the methods also
        #and then iterate over the `options` dictionary keys...
        commands = ['show_deps', 'query_metadata_pypi', 'fetch',
                'versions_available', 'show_updates', 'browse_website',
                'show_download_links', 'pypi_search', 'show_pypi_changelog',
                'show_pypi_releases', 'yolk_version', 'show_all',
                'show_active', 'show_non_active', 'show_entry_map',
                'show_entry_points']

        #Run first command it finds, and only the first command, then return
        #XXX: Check if more than one command was set in options and give error?
        for action in commands:
            if getattr(self.options, action):
                return getattr(self, action)()
        opt_parser.print_help()