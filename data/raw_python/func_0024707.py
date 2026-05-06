def _validate_other_add_sub(self, other):
        """Conditions for other to satisfy before add/sub."""
        if not isinstance(other, self.__class__):
            raise exceptions.IncompatibleSources(
                'Can only operate on {0}.'.format(self.__class__.__name__))