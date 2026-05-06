def load_yaml_by_relpath(cls, directories, rel_path, log_debug=False):
        """Load a yaml file with path that is relative to one of given directories.

        Args:
            directories: list of directories to search
            name: relative path of the yaml file to load
            log_debug: log all messages as debug
        Returns:
            tuple (fullpath, loaded yaml structure) or None if not found
        """
        for d in directories:
            if d.startswith(os.path.expanduser('~')) and not os.path.exists(d):
                os.makedirs(d)
            possible_path = os.path.join(d, rel_path)
            if os.path.exists(possible_path):
                loaded = cls.load_yaml_by_path(possible_path, log_debug=log_debug)
                if loaded is not None:
                    return (possible_path, cls.load_yaml_by_path(possible_path))

        return None