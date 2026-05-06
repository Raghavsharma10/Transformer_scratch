def row(self):
        """Return the row of the child."""
        if self.parent is not None:
            children = self.parent.getChildren()
            # The index method of the list object.
            return children.index(self)
        else:
            return 0