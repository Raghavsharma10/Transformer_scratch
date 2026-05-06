def set_data(self, index, value):
        """Uses given data setter, and emit modelReset signal"""
        acces, field = self.get_item(index), self.header[index.column()]
        self.beginResetModel()
        self.set_data_hook(acces, field, value)
        self.endResetModel()