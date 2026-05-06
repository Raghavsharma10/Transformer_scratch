def set_by_Id(self, Id, is_added):
        """Update selected_ids with given Id"""
        row = self.collection.index_from_id(Id)
        if row is None:
            return
        self._set_id(Id, is_added, self.index(row, 0))