def find_module(cls, fullname, path=None):
        """find the module on sys.path or 'path' based on sys.path_hooks and
        sys.path_importer_cache.
        This method is for python2 only
        """
        spec = cls.find_spec(fullname, path)
        if spec is None:
            return None
        elif spec.loader is None and spec.submodule_search_locations:
            # Here we need to create a namespace loader to handle namespaces since python2 doesn't...
            return NamespaceLoader2(spec.name, spec.submodule_search_locations)
        else:
            return spec.loader