def get_cache(self):
        """
        Get a package name list from disk cache or PyPI
        """
        #This is used by external programs that import `CheeseShop` and don't
        #want a cache file written to ~/.pypi and query PyPI every time.
        if self.no_cache:
            self.pkg_list = self.list_packages()
            return

        if not os.path.exists(self.yolk_dir):
            os.mkdir(self.yolk_dir)
        if os.path.exists(self.pkg_cache_file):
            self.pkg_list = self.query_cached_package_list()
        else:
            self.logger.debug("DEBUG: Fetching package list cache from PyPi...")
            self.fetch_pkg_list()