def update_cached_fields_pre_save(self, update_fields: list):
        """
        Call on pre_save signal for objects (to automatically refresh on save).
        :param update_fields: list of fields to update
        """
        if self.id and update_fields is None:
            self.update_cached_fields(commit=False, exceptions=False)