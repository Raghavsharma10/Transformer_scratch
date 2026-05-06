def get_authorizations_for_agent_and_function(self, agent_id, function_id):
        """Gets a list of ``Authorizations`` associated with a given agent.

        Authorizations related to the given resource, including those
        related through an ``Agent,`` are returned. In plenary mode, the
        returned list contains all known authorizations or an error
        results. Otherwise, the returned list may contain only those
        authorizations that are accessible through this session.

        arg:    agent_id (osid.id.Id): an agent ``Id``
        arg:    function_id (osid.id.Id): a function ``Id``
        return: (osid.authorization.AuthorizationList) - the returned
                ``Authorization list``
        raise:  NullArgument - ``agent_id`` or ``function_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        collection = JSONClientValidated('authorization',
                                         collection='Authorization',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'agentId': str(agent_id),
                  'functionId': str(function_id)},
                 **self._view_filter())).sort('_sort_id', ASCENDING)
        return objects.AuthorizationList(result, runtime=self._runtime)