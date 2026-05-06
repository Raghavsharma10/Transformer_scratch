def unassign_grade_system_from_gradebook(self, grade_system_id, gradebook_id):
        """Removes a ``GradeSystem`` from a ``Gradebook``.

        arg:    grade_system_id (osid.id.Id): the ``Id`` of the
                ``GradeSystem``
        arg:    gradebook_id (osid.id.Id): the ``Id`` of the
                ``Gradebook``
        raise:  NotFound - ``grade_system_id`` or ``gradebook_id`` not
                found or ``grade_system_id`` not assigned to
                ``gradebook_id``
        raise:  NullArgument - ``grade_system_id`` or ``gradebook_id``
                is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinAssignmentSession.unassign_resource_from_bin
        mgr = self._get_provider_manager('GRADING', local=True)
        lookup_session = mgr.get_gradebook_lookup_session(proxy=self._proxy)
        lookup_session.get_gradebook(gradebook_id)  # to raise NotFound
        self._unassign_object_from_catalog(grade_system_id, gradebook_id)