def get(self, name, *default):
        # type: (str, Any) -> Any
        """ Get context value with the given name and optional default.

        Args:
            name (str):
                The name of the context value.
            *default (Any):
                If given and the key doesn't not exist, this will be returned
                instead. If it's not given and the context value does not exist,
                `AttributeError` will be raised

        Returns:
            The requested context value.  If the value does not exist it will
            return `default` if give or raise `AttributeError`.

        Raises:
            AttributeError: If the value does not exist and `default` was not
                given.
        """

        curr = self.values
        for part in name.split('.'):
            if part in curr:
                curr = curr[part]
            elif default:
                return default[0]
            else:
                fmt = "Context value '{}' does not exist:\n{}"
                raise AttributeError(fmt.format(
                    name, util.yaml_dump(self.values)
                ))

        return curr