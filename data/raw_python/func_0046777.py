def get_item_ids(self):
        """get item ids associated with this assessment part"""
        if self.has_item_ids():
            return IdList(self.my_osid_object._my_map['itemIds'],
                          runtime=self.my_osid_object._runtime,
                          proxy=self.my_osid_object._proxy)
        raise IllegalState()