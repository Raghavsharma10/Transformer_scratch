def column(self):
        """Return the ambry column"""
        from ambry.orm.exc import NotFoundError

        if not hasattr(self, 'partition'):
            return None

        if not self.name:
            return None

        try:
            try:
                return self.partition.column(self.name)
            except AttributeError:
                return self.partition.table.column(self.name)
        except NotFoundError:
            return None