def get_items(self):
        """This is out of spec, but required for adaptive assessment parts?"""
        ils = get_item_lookup_session(runtime=self._runtime, proxy=self._proxy)
        ils.use_federated_bank_view()
        items = []
        if self.has_items():
            for idstr in self._my_map['itemIds']:
                items.append(ils.get_item(Id(idstr)))
        return ItemList(items, runtime=self._runtime, proxy=self._proxy)