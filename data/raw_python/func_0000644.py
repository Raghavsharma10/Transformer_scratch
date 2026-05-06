def restore_row(self, row, schema):
        """Restore row from SQL
        """
        row = list(row)
        for index, field in enumerate(schema.fields):
            if self.__dialect == 'postgresql':
                if field.type in ['array', 'object']:
                    continue
            row[index] = field.cast_value(row[index])
        return row