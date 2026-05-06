def import_deleted_fields(self, data):
        """
        Set data fields to deleted
        """

        def child_delete_from_str(data_str):
            """
            Inner function to set children fields to deleted
            """
            parts = data_str.split('.', 1)
            if parts[0].isnumeric:
                self[int(parts[0])].import_deleted_fields(parts[1])

        if not self.get_read_only() or not self.is_locked():
            if isinstance(data, str):
                data = [data]
            if isinstance(data, list):
                for key in data:
                    child_delete_from_str(key)