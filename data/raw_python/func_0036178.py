def traverse_path_for_valid_application_paths(self, top_path):
        """
        For every path beneath top path that does not contain the exclude
        pattern look for python, mel and images and place them in their
        corresponding system environments.
        """
        self.put_path(Path(top_path))
        for p in self._walk(top_path):
            self.put_path(p)