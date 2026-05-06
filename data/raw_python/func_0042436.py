def as_dict(self):
        """
        Convert this ObservatoryMetadataSearch to a dict, ready for serialization to JSON for use in the API.

        :return:
            Dict representation of this ObservatoryMetadataSearch instance
        """
        d = {}
        _add_value(d, 'obstory_ids', self.obstory_ids)
        _add_string(d, 'field_name', self.field_name)
        _add_value(d, 'lat_min', self.lat_min)
        _add_value(d, 'lat_max', self.lat_max)
        _add_value(d, 'long_min', self.long_min)
        _add_value(d, 'long_max', self.long_max)
        _add_value(d, 'time_min', self.time_min)
        _add_value(d, 'time_max', self.time_max)
        _add_string(d, 'item_id', self.item_id)
        _add_value(d, 'skip', self.skip)
        _add_value(d, 'limit', self.limit)
        _add_boolean(d, 'exclude_imported', self.exclude_imported)
        _add_string(d, 'exclude_export_to', self.exclude_export_to)
        return d