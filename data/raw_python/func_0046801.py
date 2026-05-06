def get_objectives_by_genus_type(self, objective_genus_type=None):
        """Gets an ObjectiveList corresponding to the given objective genus
        Type which does not include objectives of genus types derived
        from the specified Type.
        In plenary mode, the returned list contains all known objectives
        or an error results. Otherwise, the returned list may contain
        only those objectives that are accessible through this session.
        arg:    objectiveGenusType (osid.type.Type): an objective genus
                type
        return: (osid.learning.ObjectiveList) - the returned Objective
                list
        raise:  NullArgument - objectiveGenusType is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        compliance: mandatory - This method must be implemented.

        """
        if objective_genus_type is None:
            raise NullArgument()
        url_path = construct_url('objectives_by_genus',
                                 bank_id=self._catalog_idstr,
                                 genus_type=str(objective_genus_type))
        return objects.ObjectiveList(self._get_request(url_path))