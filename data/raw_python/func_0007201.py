def setData(self, index: QModelIndex, value, role=None):
        """Update selected_ids on click on index cell."""
        if not (index.isValid() and role == Qt.CheckStateRole):
            return False
        c_id = self.get_item(index).Id
        self._set_id(c_id, value == Qt.Checked, index)
        return True