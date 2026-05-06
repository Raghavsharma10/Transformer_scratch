def as_dict(self):
        """
        Convert this FileRecordSearch to a dict, ready for serialization to JSON for use in the API.

        :return:
            Dict representation of this FileRecordSearch instance
        """
        d = {}
        _add_value(d, 'obstory_ids', self.obstory_ids)
        _add_value(d, 'lat_min', self.lat_min)
        _add_value(d, 'lat_max', self.lat_max)
        _add_value(d, 'long_min', self.long_min)
        _add_value(d, 'long_max', self.long_max)
        _add_value(d, 'time_min', self.time_min)
        _add_value(d, 'time_max', self.time_max)
        _add_value(d, 'mime_type', self.mime_type)
        _add_value(d, 'skip', self.skip)
        _add_value(d, 'limit', self.limit)
        _add_string(d, 'semantic_type', self.semantic_type)
        _add_string(d, 'observation_type', self.observation_type)
        _add_value(d, 'observation_id', self.observation_id)
        _add_string(d, 'repository_fname', self.repository_fname)
        _add_boolean(d, 'exclude_imported', self.exclude_imported)
        _add_string(d, 'exclude_export_to', self.exclude_export_to)
        d['meta'] = list((x.as_dict() for x in self.meta_constraints))
        return d