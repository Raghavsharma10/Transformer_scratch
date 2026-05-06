def add_child_objective(self, objective_id, child_id):
        """Adds a child to an objective.

        arg:    objective_id (osid.id.Id): the ``Id`` of an objective
        arg:    child_id (osid.id.Id): the ``Id`` of the new child
        raise:  AlreadyExists - ``objective_id`` is already a parent of
                ``child_id``
        raise:  NotFound - ``objective_id`` or ``child_id`` not found
        raise:  NullArgument - ``objective_id`` or ``child_id`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.ontology.SubjectHierarchyDesignSession.add_child_subject_template
        return self._hierarchy_session.add_child(id_=objective_id, child_id=child_id)