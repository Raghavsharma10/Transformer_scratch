def _set_id(self, Id, is_added, index):
        """Update selected_ids and emit dataChanged"""
        if is_added:
            self.selected_ids.add(Id)
        else:
            self.selected_ids.remove(Id)
        self.dataChanged.emit(index, index)