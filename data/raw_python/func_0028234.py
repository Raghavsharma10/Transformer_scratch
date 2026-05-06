def _find_watch_paths(self):
        """
        Find paths to watch.

        :return:
            Paths to watch.
        """
        # Add directory paths in `sys.path` to watch paths
        watch_path_s = set(os.path.abspath(x) for x in sys.path)

        # For each extra path
        for extra_path in self._extra_paths or ():
            # Get the extra path's directory path
            extra_dir_path = os.path.dirname(os.path.abspath(extra_path))

            # Add to watch paths
            watch_path_s.add(extra_dir_path)

        # For each module in `sys.modules`
        for module in list(sys.modules.values()):
            # Get module file path
            module_path = getattr(module, '__file__', None)

            # If have module file path
            if module_path is not None:
                # Get module directory path
                module_dir_path = os.path.dirname(os.path.abspath(module_path))

                # Add to watch paths
                watch_path_s.add(module_dir_path)

        # Find short paths of these watch paths.
        # E.g. if both `/home` and `/home/aoik` exist, only keep `/home`.
        watch_path_s = self._find_short_paths(watch_path_s)

        # Return the watch paths
        return watch_path_s