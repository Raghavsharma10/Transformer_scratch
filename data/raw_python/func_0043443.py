def get_attrs_by_path(self, field_path, stop_first=False):
        """
        It returns list of values looked up by field path.
        Field path is dot-formatted string path: ``parent_field.child_field``.

        :param field_path: field path. It allows ``*`` as wildcard.
        :type field_path: list or None.
        :param stop_first: Stop iteration on first value looked up. Default: False.
        :type stop_first: bool
        :return: A list of values or None it was a invalid path.
        :rtype: :class:`list` or :class:`None`
        """
        fields, next_field = self._get_fields_by_path(field_path)
        values = []
        for field in fields:
            if next_field:
                try:
                    res = self.get_field_value(field).get_attrs_by_path(next_field, stop_first=stop_first)
                    if res is None:
                        continue
                    values.extend(res)

                    if stop_first and len(values):
                        break

                except AttributeError:
                    pass
            else:
                value = self.get_field_value(field)
                if value is None:
                    continue
                if stop_first:
                    return [value, ]
                values.append(value)

        return values if len(values) else None