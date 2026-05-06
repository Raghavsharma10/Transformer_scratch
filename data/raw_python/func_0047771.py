def get_resource_by_agent(self, agent_id):
        """Gets the ``Resource`` associated with the given agent.

        arg:    agent_id (osid.id.Id): ``Id`` of the ``Agent``
        return: (osid.resource.Resource) - associated resource
        raise:  NotFound - ``agent_id`` is not found
        raise:  NullArgument - ``agent_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        collection = JSONClientValidated('resource',
                                         collection='Resource',
                                         runtime=self._runtime)
        result = collection.find_one(
            dict({'agentIds': {'$in': [str(agent_id)]}},
                 **self._view_filter()))
        return objects.Resource(
            osid_object_map=result,
            runtime=self._runtime,
            proxy=self._proxy)