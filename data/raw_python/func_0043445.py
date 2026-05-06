def delete_attr_by_path(self, field_path):
        """
        It deletes fields looked up by field path.
        Field path is dot-formatted string path: ``parent_field.child_field``.

        :param field_path: field path. It allows ``*`` as wildcard.
        :type field_path: str
        """
        fields, next_field = self._get_fields_by_path(field_path)
        for field in fields:
            if next_field:
                try:
                    self.get_field_value(field).delete_attr_by_path(next_field)
                except AttributeError:
                    pass
            else:
                self.delete_field_value(field)