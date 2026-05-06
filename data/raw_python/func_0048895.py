def get_agent(self, agent_id):
        """Gets the ``Agent`` specified by its ``Id``.

        In plenary mode, the exact ``Id`` is found or a ``NotFound``
        results. Otherwise, the returned ``Agent`` may have a different
        ``Id`` than requested, such as the case where a duplicate ``Id``
        was assigned to an ``Agent`` and retained for compatibility.

        arg:    agent_id (osid.id.Id): the ``Id`` of an ``Agent``
        return: (osid.authentication.Agent) - the returned ``Agent``
        raise:  NotFound - no ``Agent`` found with the given ``Id``
        raise:  NullArgument - ``agent_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resource
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('authentication',
                                         collection='Agent',
                                         runtime=self._runtime)
        result = collection.find_one(
            dict({'_id': ObjectId(self._get_id(agent_id, 'authentication').get_identifier())},
                 **self._view_filter()))
        return objects.Agent(osid_object_map=result, runtime=self._runtime, proxy=self._proxy)