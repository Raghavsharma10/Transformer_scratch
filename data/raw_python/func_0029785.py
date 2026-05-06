def is_empty(self):
        """Return True if the table has no columns or the only column is the id"""
        if len(self.columns) == 0:
            return True

        if len(self.columns) == 1 and self.columns[0].name == 'id':
            return True

        return False