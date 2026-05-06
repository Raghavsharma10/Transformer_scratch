def bind_to(self, table, name):
        """
        Bind this column to a table, and assign it a name. This method
        can only be called once per instance, because a Column cannot be
        bound to multiple tables. (The sort order would be ambiguous.)
        """
        if self.bound_to is not None:
            raise AttributeError(
                "Column is already bound to '%s' as '%s'" %\
                    self.bound_to)

        self.bound_to = (table, name)