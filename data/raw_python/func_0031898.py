def same_kind(self, other):
        """
        Return True if "other" is an object of the same type and it was
        instantiated with the same parameters
        """

        return type(self) is type(other) and self._same_parameters(other)