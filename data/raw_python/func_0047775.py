def unassign_agent_from_resource(self, agent_id, resource_id):
        """Removes an ``Agent`` from a ``Resource``.

        arg:    agent_id (osid.id.Id): the ``Id`` of the ``Agent``
        arg:    resource_id (osid.id.Id): the ``Id`` of the ``Resource``
        raise:  NotFound - ``agent_id`` or ``resource_id`` not found or
                ``agent_id`` not assigned to ``resource_id``
        raise:  NullArgument - ``agent_id`` or ``resource_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        collection = JSONClientValidated('resource',
                                         collection='Resource',
                                         runtime=self._runtime)
        resource = collection.find_one({'_id': ObjectId(resource_id.get_identifier())})

        try:
            resource['agentIds'].remove(str(agent_id))
        except (KeyError, ValueError):
            raise errors.NotFound('agent_id not assigned to resource')
        collection.save(resource)