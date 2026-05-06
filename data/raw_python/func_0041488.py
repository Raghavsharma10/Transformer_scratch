def find_module(self, fullname):
        """Try to find a loader for the specified module, or the namespace
        package portions. Returns loader.
        """

        spec = self.find_spec(fullname)
        if spec is None:
            return None

        # We need to handle the namespace case here for python2
        if spec.loader is None and len(spec.submodule_search_locations):
            spec.loader = NamespaceLoader2(spec.name, spec.submodule_search_locations)

        return spec.loader