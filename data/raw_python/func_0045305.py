def get_unpermitted_fields(self):
        """
        Gives unpermitted fields for current context/user.

        Returns:
            List of unpermitted field names.
        """
        return (self._unpermitted_fields if self._is_unpermitted_fields_set else
                self._apply_cell_filters(self._context))