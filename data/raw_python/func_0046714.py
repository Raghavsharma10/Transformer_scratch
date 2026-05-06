def assign_grade_system_to_gradebook(self, grade_system_id, gradebook_id):
        """Adds an existing ``GradeSystem`` to a ``Gradebook``.

        arg:    grade_system_id (osid.id.Id): the ``Id`` of the
                ``GradeSystem``
        arg:    gradebook_id (osid.id.Id): the ``Id`` of the
                ``Gradebook``
        raise:  AlreadyExists - ``grade_system_id`` is already assigned
                to ``gradebook_id``
        raise:  NotFound - ``grade_system_id`` or ``gradebook_id`` not
                found
        raise:  NullArgument - ``grade_system_id`` or ``gradebook_id``
                is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinAssignmentSession.assign_resource_to_bin
        mgr = self._get_provider_manager('GRADING', local=True)
        lookup_session = mgr.get_gradebook_lookup_session(proxy=self._proxy)
        lookup_session.get_gradebook(gradebook_id)  # to raise NotFound
        self._assign_object_to_catalog(grade_system_id, gradebook_id)