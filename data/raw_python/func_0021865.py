def fetch_pkg_list(self):
        """Fetch and cache master list of package names from PYPI"""
        self.logger.debug("DEBUG: Fetching package name list from PyPI")
        package_list = self.list_packages()
        cPickle.dump(package_list, open(self.pkg_cache_file, "w"))
        self.pkg_list = package_list