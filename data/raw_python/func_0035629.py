def add_column_property_xsd(self, tb, column_property):
        """ Add the XSD for a column property to the ``TreeBuilder``. """
        if len(column_property.columns) != 1:
            raise NotImplementedError  # pragma: no cover
        column = column_property.columns[0]
        if column.primary_key and not self.include_primary_keys:
            return
        if column.foreign_keys and not self.include_foreign_keys:
            if len(column.foreign_keys) != 1:  # pragma: no cover
                # FIXME understand when a column can have multiple
                # foreign keys
                raise NotImplementedError()
            return
        attrs = {'name': column_property.key}
        self.add_column_xsd(tb, column, attrs)