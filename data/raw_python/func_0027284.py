def _make_value(self, value):
        """Instantiates an enum with an arbitrary value."""
        member = self.__new__(self, value)
        member.__init__(value)
        return member