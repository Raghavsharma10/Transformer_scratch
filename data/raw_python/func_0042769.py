def _get_generator(self, name):
        """Load the generator plugin and execute its lifecycle.

        :param dist: distribution
        """
        for ep in pkg_resources.iter_entry_points(self.group, name=None):
            if ep.name == name:
                generator = ep.load()
                return generator