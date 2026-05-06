def qualname(self) -> str:
        """
        Returns the fully qualified name of the class-under-construction, if possible,
        otherwise just the class name.
        """
        if self.module:
            return self.module + '.' + self.name
        return self.name