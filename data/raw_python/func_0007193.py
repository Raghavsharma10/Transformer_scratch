def sort(self, section: int, order=None):
        """Order is defined by the current state of sorting"""
        attr = self.header[section]
        old_i, old_sort = self.sort_state
        self.beginResetModel()
        if section == old_i:
            self.collection.sort(attr, not old_sort)
            self.sort_state = (section, not old_sort)
        else:
            self.collection.sort(attr, True)
            self.sort_state = (section, True)
        self.endResetModel()