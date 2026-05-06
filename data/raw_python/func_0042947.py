def put(self, *items) -> "AttrIndexedDict":
        "Add items to the dict that will be indexed by self.attr."
        for item in items:
            self.data[getattr(item, self.attr)] = item
        return self