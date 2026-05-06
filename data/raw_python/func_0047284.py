def get_item_ids(self):
        """This is out of spec, but required for adaptive assessment parts?"""
        item_ids = []
        if self.has_items():
            for idstr in self._my_map['itemIds']:
                item_ids.append(idstr)
        return IdList(item_ids)