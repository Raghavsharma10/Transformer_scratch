def change_history_fields(self, fields, value=None):
        r"""

        """
        if not isinstance(fields, list):
            raise Exception('fields should be a list')

        self._change_history['fields'] = fields
        if value:
            self._change_history['value'] = value

        return self