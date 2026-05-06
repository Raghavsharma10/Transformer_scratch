def restore_row(self, row, schema, pk):
        """Restore row from Pandas
        """
        result = []
        for field in schema.fields:
            if schema.primary_key and schema.primary_key[0] == field.name:
                if field.type == 'number' and np.isnan(pk):
                    pk = None
                if pk and field.type == 'integer':
                    pk = int(pk)
                result.append(field.cast_value(pk))
            else:
                value = row[field.name]
                if field.type == 'number' and np.isnan(value):
                    value = None
                if value and field.type == 'integer':
                    value = int(value)
                elif field.type == 'datetime':
                    value = value.to_pydatetime()
                result.append(field.cast_value(value))
        return result