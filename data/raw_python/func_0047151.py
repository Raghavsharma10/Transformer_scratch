def get_child_objectives(self, objective_id):
        """Gets the children of the given objective.

        arg:    objective_id (osid.id.Id): the ``Id`` to query
        return: (osid.learning.ObjectiveList) - the children of the
                objective
        raise:  NotFound - ``objective_id`` is not found
        raise:  NullArgument - ``objective_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.ontology.SubjectHierarchySession.get_child_subjects_template
        if self._hierarchy_session.has_children(objective_id):
            child_ids = self._hierarchy_session.get_children(objective_id)
            collection = JSONClientValidated('learning',
                                             collection='Objective',
                                             runtime=self._runtime)
            result = collection.find(
                dict({'_id': {'$in': [ObjectId(child_id.get_identifier()) for child_id in child_ids]}},
                     **self._view_filter()))
            return objects.ObjectiveList(
                result,
                runtime=self._runtime,
                proxy=self._proxy)
        raise errors.IllegalState('no children')