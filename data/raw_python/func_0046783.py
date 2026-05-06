def get_item_ids_metadata(self):
        """get the metadata for item"""
        metadata = dict(self._item_ids_metadata)
        metadata.update({'existing_id_values': self.my_osid_object_form._my_map['itemIds']})
        return Metadata(**metadata)