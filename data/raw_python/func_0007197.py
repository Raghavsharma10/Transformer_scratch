def set_collection(self, collection):
        """Reset sort state, set collection and emit resetModel signal"""
        self.beginResetModel()
        self.collection = collection
        self.sort_state = (-1, False)
        self.endResetModel()