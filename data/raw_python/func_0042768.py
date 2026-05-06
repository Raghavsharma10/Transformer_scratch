def _get_generators(self):
        """Get installed banana plugins.

        :return: dictionary of installed generators name: distribution
        """
        # on using entrypoints:
        # http://stackoverflow.com/questions/774824/explain-python-entry-points
        # TODO: make sure we do not have conflicting generators installed!
        generators = [ep.name for ep in
                      pkg_resources.iter_entry_points(self.group)]
        return generators