def as_dict(self):
        """
        Convert this ObservationGroupSearch to a dict, ready for serialization to JSON for use in the API.

        :return:
            Dict representation of this ObservationGroupSearch instance
        """
        d = {}
        _add_string(d, 'obstory_name', self.obstory_name)
        _add_string(d, 'semantic_type', self.semantic_type)
        _add_value(d, 'time_min', self.time_min)
        _add_value(d, 'time_max', self.time_max)
        _add_string(d, 'group_id', self.group_id)
        _add_string(d, 'observation_id', self.observation_id)
        _add_value(d, 'skip', self.skip)
        _add_value(d, 'limit', self.limit)
        d['meta'] = list((x.as_dict() for x in self.meta_constraints))
        return d