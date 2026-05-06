def _transform_row(self, row):
        """Transforms a Row into Python values.

        :param row:
            A ``common_pb2.Row`` object.

        :returns:
            A list of values casted into the correct Python types.

        :raises:
            NotImplementedError
        """
        tmp_row = []

        for i, column in enumerate(row.value):
            if column.has_array_value:
                raise NotImplementedError('array types are not supported')
            elif column.scalar_value.null:
                tmp_row.append(None)
            else:
                field_name, rep, mutate_to, cast_from = self._column_data_types[i]

                # get the value from the field_name
                value = getattr(column.scalar_value, field_name)

                # cast the value
                if cast_from is not None:
                    value = cast_from(value)

                tmp_row.append(value)
        return tmp_row