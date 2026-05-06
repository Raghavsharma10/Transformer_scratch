def remove_line(self, section):
        """Base implementation just pops the item from collection.
        Re-implements to add global behaviour
        """
        self.beginResetModel()
        self.collection.pop(section)
        self.endResetModel()