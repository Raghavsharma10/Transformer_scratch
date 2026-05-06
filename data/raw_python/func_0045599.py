def memory_usage(self):
        """
        Get the combined memory usage of the field data and field values.
        """
        data = super(Field, self).memory_usage()
        values = 0
        for value in self.field_values:
            values += value.memory_usage()
        data['field_values'] = values
        return data