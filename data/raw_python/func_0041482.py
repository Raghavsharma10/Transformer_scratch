def _path_importer_cache(cls, path):  # from importlib.PathFinder
        """Get the finder for the path entry from sys.path_importer_cache.
        If the path entry is not in the cache, find the appropriate finder
        and cache it. If no finder is available, store None.
        """
        if path == '':
            try:
                path = os.getcwd()
            except FileNotFoundError:
                # Don't cache the failure as the cwd can easily change to
                # a valid directory later on.
                return None
        try:
            finder = sys.path_importer_cache[path]
        except KeyError:
            finder = cls._path_hooks(path)
            sys.path_importer_cache[path] = finder

        return finder