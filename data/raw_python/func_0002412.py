def update_existing_pivot(self, id, attributes, touch=True):
        """
        Update an existing pivot record on the table.
        """
        if self.updated_at() in self._pivot_columns:
            attributes = self.set_timestamps_on_attach(attributes, True)

        updated = self._new_picot_statement_for_id(id).update(attributes)

        if touch:
            self.touch_if_touching()

        return updated