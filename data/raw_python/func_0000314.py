def get_column_name(self, column_name):
        """ Get a column for given column name from META api. """
        name = pretty_name(column_name)
        if column_name in self._meta.columns:
            column_cls = self._meta.columns[column_name]
            if column_cls.verbose_name:
                name = column_cls.verbose_name
        return name