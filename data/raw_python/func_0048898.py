def get_agents(self):
        """Gets all ``Agents``.

        In plenary mode, the returned list contains all known agents or
        an error results. Otherwise, the returned list may contain only
        those agents that are accessible through this session.

        return: (osid.authentication.AgentList) - a list of ``Agents``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('authentication',
                                         collection='Agent',
                                         runtime=self._runtime)
        result = collection.find(self._view_filter()).sort('_id', DESCENDING)
        return objects.AgentList(result, runtime=self._runtime, proxy=self._proxy)