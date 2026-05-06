def get_allow_repeat_items_metadata(self):
        """get the metadata for allow repeat items"""
        metadata = dict(self._allow_repeat_items_metadata)
        metadata.update({'existing_id_values': self.my_osid_object_form._my_map['allowRepeatItems']})
        return Metadata(**metadata)