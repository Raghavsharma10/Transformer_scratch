def _update(self):
        """Emit dataChanged signal on all cells"""
        self.dataChanged.emit(self.createIndex(0, 0), self.createIndex(
            len(self.collection), len(self.header)))