def insert(self, i, v):
        """Insert an idfobject (bunch) to list1 and its object to list2."""
        self.list1.insert(i, v)
        self.list2.insert(i, v.obj)
        if isinstance(v, EpBunch):
            v.theidf = self.theidf