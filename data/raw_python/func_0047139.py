def get_objective(self, objective_id):
        """Gets the ``Objective`` specified by its ``Id``.

        In plenary mode, the exact ``Id`` is found or a ``NotFound``
        results. Otherwise, the returned ``Objective`` may have a
        different ``Id`` than requested, such as the case where a
        duplicate ``Id`` was assigned to an ``Objective`` and retained
        for compatibility.

        arg:    objective_id (osid.id.Id): ``Id`` of the ``Objective``
        return: (osid.learning.Objective) - the objective
        raise:  NotFound - ``objective_id`` not found
        raise:  NullArgument - ``objective_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method is must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceLookupSession.get_resource
        # NOTE: This implementation currently ignores plenary view
        collection = JSONClientValidated('learning',
                                         collection='Objective',
                                         runtime=self._runtime)
        result = collection.find_one(
            dict({'_id': ObjectId(self._get_id(objective_id, 'learning').get_identifier())},
                 **self._view_filter()))
        return objects.Objective(osid_object_map=result, runtime=self._runtime, proxy=self._proxy)