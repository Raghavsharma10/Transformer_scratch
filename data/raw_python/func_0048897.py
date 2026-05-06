def get_agents_by_genus_type(self, agent_genus_type):
        """Gets an ``AgentList`` corresponding to the given agent genus ``Type`` which does not include agents of genus types derived from the specified ``Type``.

        In plenary mode, the returned list contains all known agents or
        an error results. Otherwise, the returned list may contain only
        those agents that are accessible through this session.

        arg:    agent_genus_type (osid.type.Type): an agent genus type
        return: (osid.authentication.AgentList) - the returned ``Agent``
                list
        raise:  NullArgument - ``agent_genus_type`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resources_by_genus_type
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('authentication',
                                         collection='Agent',
                                         runtime=self._runtime)
        result = collection.find(
            dict({'genusTypeId': str(agent_genus_type)},
                 **self._view_filter())).sort('_id', DESCENDING)
        return objects.AgentList(result, runtime=self._runtime, proxy=self._proxy)