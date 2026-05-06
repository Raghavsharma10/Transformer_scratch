def set(self, name, value):
        """ Set context value.

        Args:
            name (str):
                The name of the context value to change.
            value (Any):
                The new value for the selected context value
        """
        curr = self.values
        parts = name.split('.')

        for i, part in enumerate(parts[:-1]):
            try:
                curr = curr.setdefault(part, {})
            except AttributeError:
                raise InvalidPath('.'.join(parts[:i + 1]))

        try:
            curr[parts[-1]] = value
        except TypeError:
            raise InvalidPath('.'.join(parts[:-1]))