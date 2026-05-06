def get_1st_attr_by_path(self, field_path, **kwargs):
        """
        It returns first value looked up by field path.
        Field path is dot-formatted string path: ``parent_field.child_field``.

        :param field_path: field path. It allows ``*`` as wildcard.
        :type field_path: str
        :param default: Default value if field does not exist.
                        If it is not defined :class:`AttributeError` exception will be raised.
        :return: value
        """

        res = self.get_attrs_by_path(field_path, stop_first=True)
        if res is None:
            try:
                return kwargs['default']
            except KeyError:
                raise AttributeError("Field '{0}' does not exist".format(field_path))
        return res.pop()