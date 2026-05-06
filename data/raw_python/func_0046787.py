def get_max_levels_metadata(self):
        """get the metadata for max levels"""
        metadata = dict(self._max_levels_metadata)
        metadata.update({'existing_cardinal_values': self.my_osid_object_form._my_map['maxLevels']})
        return Metadata(**metadata)