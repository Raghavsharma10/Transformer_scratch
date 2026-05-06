def remove_child_objective(self, objective_id, child_id):
        """Removes a child from an objective.

        arg:    objective_id (osid.id.Id): the ``Id`` of an objective
        arg:    child_id (osid.id.Id): the ``Id`` of the new child
        raise:  NotFound - ``objective_id`` not a parent of ``child_id``
        raise:  NullArgument - ``objective_id`` or ``child_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.ontology.SubjectHierarchyDesignSession.remove_child_subject_template
        return self._hierarchy_session.remove_child(id_=objective_id, child_id=child_id)