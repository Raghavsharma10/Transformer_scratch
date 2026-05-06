def maxId(self):
        """int: current max id of objects"""
        if len(self.model.db) == 0:
            return 0

        return max(map(lambda obj: obj["id"], self.model.db))