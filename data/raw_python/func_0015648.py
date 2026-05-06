def _convert_value(self, column, value):
        '''Convert value to a GObject.Value of the expected type'''

        if isinstance(value, GObject.Value):
            return value
        return GObject.Value(self.get_column_type(column), value)