def _apply_cell_filters(self, context):
        """
        Applies the field restrictions based on the
         return value of the context's "has_permission()" method.
         Stores them on self._unpermitted_fields.

        Returns:
            List of unpermitted fields names.
        """
        self.setattrs(_is_unpermitted_fields_set=True)
        for perm, fields in self.Meta.field_permissions.items():
            if not context.has_permission(perm):
                self._unpermitted_fields.extend(fields)
        return self._unpermitted_fields