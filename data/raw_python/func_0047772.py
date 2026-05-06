def get_agent_ids_by_resource(self, resource_id):
        """Gets the list of ``Agent``  ``Ids`` mapped to a ``Resource``.

        arg:    resource_id (osid.id.Id): ``Id`` of a ``Resource``
        return: (osid.id.IdList) - list of agent ``Ids``
        raise:  NotFound - ``resource_id`` is not found
        raise:  NullArgument - ``resource_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        collection = JSONClientValidated('resource',
                                         collection='Resource',
                                         runtime=self._runtime)
        resource = collection.find_one(
            dict({'_id': ObjectId(resource_id.get_identifier())},
                 **self._view_filter()))
        if 'agentIds' not in resource:
            result = IdList([])
        else:
            result = IdList(resource['agentIds'])
        return result