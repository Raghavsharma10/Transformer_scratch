def _get_instance_path(self, name):
        "Return a path to the pickled data with key ``name``."
        fname = self.pattern.format(**{'name': name})
        logger.debug(f'path {self.create_path}: {self.create_path.exists()}')
        self._create_path_dir()
        return Path(self.create_path, fname)