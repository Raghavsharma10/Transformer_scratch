def from_dict(d):
        """
        Builds a new instance of FileRecordSearch from a dict

        :param Object d: the dict to parse
        :return: a new FileRecordSearch based on the supplied dict
        """
        obstory_ids = _value_from_dict(d, 'obstory_ids')
        lat_min = _value_from_dict(d, 'lat_min')
        lat_max = _value_from_dict(d, 'lat_max')
        long_min = _value_from_dict(d, 'long_min')
        long_max = _value_from_dict(d, 'long_max')
        time_min = _value_from_dict(d, 'time_min')
        time_max = _value_from_dict(d, 'time_max')
        mime_type = _string_from_dict(d, 'mime_type')
        skip = _value_from_dict(d, 'skip', 0)
        limit = _value_from_dict(d, 'limit', 100)
        semantic_type = _string_from_dict(d, 'semantic_type')
        observation_type = _string_from_dict(d, 'observation_type')
        observation_id = _value_from_dict(d, 'observation_id')
        repository_fname = _string_from_dict(d, 'repository_fname')
        exclude_imported = _boolean_from_dict(d, 'exclude_imported')
        exclude_export_to = _string_from_dict(d, 'exclude_export_to')
        if 'meta' in d:
            meta_constraints = list((MetaConstraint.from_dict(x) for x in d['meta']))
        else:
            meta_constraints = []
        return FileRecordSearch(obstory_ids=obstory_ids, lat_min=lat_min, lat_max=lat_max, long_min=long_min,
                                long_max=long_max, time_min=time_min, time_max=time_max, mime_type=mime_type,
                                semantic_type=semantic_type,
                                observation_type=observation_type,
                                observation_id=observation_id, repository_fname=repository_fname,
                                meta_constraints=meta_constraints, limit=limit, skip=skip,
                                exclude_imported=exclude_imported,
                                exclude_export_to=exclude_export_to)