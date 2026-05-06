def revision(self):
        """The name of the feature branch (a string)."""
        location, _, revision = self.expression.partition('#')
        return revision if location and revision else self.expression