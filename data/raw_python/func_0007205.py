def get_current_item(self):
        """Returns (first) selected item or None"""
        l = self.selectedIndexes()
        if len(l) > 0:
            return self.model().get_item(l[0])