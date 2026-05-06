def find_loader(self, fullname):
        """Try to find a loader for the specified module, or the namespace
        package portions. Returns (loader, list-of-portions).
        This method is deprecated.  Use find_spec() instead.
        """
        spec = self.find_spec(fullname)
        if spec is None:
            return None, []
        return spec.loader, spec.submodule_search_locations or []