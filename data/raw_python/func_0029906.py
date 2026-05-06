def _dict(self, with_name=True):
        """Returns the identity as a dict.

        values that are empty are removed

        """

        d = dict([(k, getattr(self, k)) for k, _, _ in self.name_parts])

        return self.clear_dict(d)