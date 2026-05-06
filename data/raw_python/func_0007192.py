def flags(self, index: QModelIndex):
        """All fields are selectable"""
        if self.IS_EDITABLE and self.header[index.column()] in self.EDITABLE_FIELDS:
            return Qt.ItemIsEnabled | Qt.ItemIsSelectable | Qt.ItemIsEditable
        else:
            return super().flags(index) | Qt.ItemIsSelectable