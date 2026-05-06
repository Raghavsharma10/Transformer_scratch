def is_changed(self, field, from_db=False):
        """
        Args:
            field (string): Field name.
            from_db (bool): Check changes against actual db data

        Returns:
            bool: True if given fields value is changed.
        """
        return field in self.changed_fields(from_db=from_db)