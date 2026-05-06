def location(self):
        """The location of the repository that contains :attr:`revision` (a string or :data:`None`)."""
        location, _, revision = self.expression.partition('#')
        return location if location and revision else None