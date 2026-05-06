def delete_objective(self, objective_id):
        """Deletes the ``Objective`` identified by the given ``Id``.

        arg:    objective_id (osid.id.Id): the ``Id`` of the
                ``Objective`` to delete
        raise:  NotFound - an ``Objective`` was not found identified by
                the given ``Id``
        raise:  NullArgument - ``objective_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.learning.ObjectiveAdminSession.delete_objective_template

        if not isinstance(objective_id, ABCId):
            raise errors.InvalidArgument('the argument is not a valid OSID Id')
        collection = JSONClientValidated('learning',
                                         collection='Activity',
                                         runtime=self._runtime)
        if collection.find({'objectiveId': str(objective_id)}).count() != 0:
            raise errors.IllegalState('there are still Activitys associated with this Objective')

        collection = JSONClientValidated('learning',
                                         collection='Objective',
                                         runtime=self._runtime)
        collection.delete_one({'_id': ObjectId(objective_id.get_identifier())})