def get_attrs_by_path(self, field_path, stop_first=False):
        """
        It returns list of values looked up by field path.
        Field path is dot-formatted string path: ``parent_field.child_field``.

        :param field_path: field path. It allows ``*`` as wildcard.
        :type field_path: list or None.
        :param stop_first: Stop iteration on first value looked up. Default: False.
        :type stop_first: bool
        :return: value
        """
        index_list, next_field = self._get_indexes_by_path(field_path)
        values = []
        for idx in index_list:
            if next_field:
                try:
                    res = self[idx].get_attrs_by_path(next_field, stop_first=stop_first)
                    if res is None:
                        continue
                    values.extend(res)

                    if stop_first and len(values):
                        break

                except AttributeError:
                    pass
            else:
                if stop_first:
                    return [self[idx], ]
                values.append(self[idx])

        return values if len(values) else None