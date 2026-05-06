def query_versions_pypi(self, package_name):
        """Fetch list of available versions for a package from The CheeseShop"""
        if not package_name in self.pkg_list:
            self.logger.debug("Package %s not in cache, querying PyPI..." \
                    % package_name)
            self.fetch_pkg_list()
        #I have to set version=[] for edge cases like "Magic file extensions"
        #but I'm not sure why this happens. It's included with Python or
        #because it has a space in it's name?
        versions = []
        for pypi_pkg in self.pkg_list:
            if pypi_pkg.lower() == package_name.lower():
                if self.debug:
                    self.logger.debug("DEBUG: %s" % package_name)
                versions = self.package_releases(pypi_pkg)
                package_name = pypi_pkg
                break
        return (package_name, versions)