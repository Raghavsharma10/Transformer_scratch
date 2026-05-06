def get_objectives_by_query(self, objective_query=None):
        """Gets a list of Objectives matching the given objective query.

        arg:    objectiveQuery (osid.learning.ObjectiveQuery): the
                objective query
        return: (osid.learning.ObjectiveList) - the returned
                ObjectiveList
        raise:  NullArgument - objectiveQuery is null
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unsupported - objectiveQuery is not of this service
        compliance: mandatory - This method must be implemented.

        """
        if objective_query is None:
            raise NullArgument()
        if 'ancestorObjectiveId' in objective_query._query_terms:
            url_path = construct_url('objectives',
                                     bank_id=self._objective_bank_id,
                                     obj_id=objective_query._query_terms['ancestorObjectiveId'].split('=')[1])
            url_path += '/children'
        elif 'descendantObjectiveId' in objective_query._query_terms:
            url_path = construct_url('objectives',
                                     bank_id=self._objective_bank_id,
                                     obj_id=objective_query._query_terms['descendantObjectiveId'].split('=')[1])
            url_path += '/parents'
        else:
            url_path = construct_url('objectives', obj_id=None)

        for term in objective_query._query_terms:
            if term not in ['ancestorObjectiveId', 'descendantObjectiveId']:
                url_path += '&{0}'.format(objective_query._query_terms[term])

        url_path = url_path.replace('&', '?', 1)
        return objects.ObjectiveList(self._get_request(url_path))