def query_cached_package_list(self):
        """Return list of pickled package names from PYPI"""
        if self.debug:
            self.logger.debug("DEBUG: reading pickled cache file")
        return cPickle.load(open(self.pkg_cache_file, "r"))