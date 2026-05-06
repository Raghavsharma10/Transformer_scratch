def show_updates(self):
        """
        Check installed packages for available updates on PyPI

        @param project_name: optional package name to check; checks every
                             installed pacakge if none specified
        @type project_name: string

        @returns: None
        """
        dists = Distributions()
        if self.project_name:
            #Check for a single package
            pkg_list = [self.project_name]
        else:
            #Check for every installed package
            pkg_list = get_pkglist()
        found = None
        for pkg in pkg_list:
            for (dist, active) in dists.get_distributions("all", pkg,
                    dists.get_highest_installed(pkg)):
                (project_name, versions) = \
                        self.pypi.query_versions_pypi(dist.project_name)
                if versions:

                    #PyPI returns them in chronological order,
                    #but who knows if its guaranteed in the API?
                    #Make sure we grab the highest version:

                    newest = get_highest_version(versions)
                    if newest != dist.version:

                        #We may have newer than what PyPI knows about

                        if pkg_resources.parse_version(dist.version) < \
                            pkg_resources.parse_version(newest):
                            found = True
                            print(" %s %s (%s)" % (project_name, dist.version,
                                    newest))
        if not found and self.project_name:
            self.logger.info("You have the latest version installed.")
        elif not found:
            self.logger.info("No newer packages found at The Cheese Shop")
        return 0