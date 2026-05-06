def assign_agent_to_resource(self, agent_id, resource_id):
        """Adds an existing ``Agent`` to a ``Resource``.

        arg:    agent_id (osid.id.Id): the ``Id`` of the ``Agent``
        arg:    resource_id (osid.id.Id): the ``Id`` of the ``Resource``
        raise:  AlreadyExists - ``agent_id`` is already assigned to
                ``resource_id``
        raise:  NotFound - ``agent_id`` or ``resource_id`` not found
        raise:  NullArgument - ``agent_id`` or ``resource_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Should check for existence of Agent? We may mever manage them.
        collection = JSONClientValidated('resource',
                                         collection='Resource',
                                         runtime=self._runtime)
        resource = collection.find_one({'_id': ObjectId(resource_id.get_identifier())})

        try:
            ResourceAgentSession(
                self._catalog_id, self._proxy, self._runtime).get_resource_by_agent(agent_id)
        except errors.NotFound:
            pass
        else:
            raise errors.AlreadyExists()
        if 'agentIds' not in resource:
            resource['agentIds'] = [str(agent_id)]
        else:
            resource['agentIds'].append(str(agent_id))
        collection.save(resource)