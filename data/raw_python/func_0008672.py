def value_to_string(self, obj):
        """Convert a field value to a string.

        Returns the state name.
        """
        statefield = self.to_python(self.value_from_object(obj))
        return statefield.state.name