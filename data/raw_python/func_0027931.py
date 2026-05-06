def tagNames(self):
        """
        Return an iterator of unicode strings - the unique tag names which have
        been applied objects in this catalog.
        """
        return self.store.query(_TagName, _TagName.catalog == self).getColumn("name")