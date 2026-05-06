def unassign_gradebook_column_from_gradebook(self, gradebook_column_id, gradebook_id):
        """Removes a ``GradebookColumn`` from a ``Gradebook``.

        arg:    gradebook_column_id (osid.id.Id): the ``Id`` of the
                ``GradebookColumn``
        arg:    gradebook_id (osid.id.Id): the ``Id`` of the
                ``Gradebook``
        raise:  NotFound - ``gradebook_column_id`` or ``gradebook_id``
                not found or ``gradebook_column_id`` not assigned to
                ``gradebook_id``
        raise:  NullArgument - ``gradebook_column_id`` or
                ``gradebook_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinAssignmentSession.unassign_resource_from_bin
        mgr = self._get_provider_manager('GRADING', local=True)
        lookup_session = mgr.get_gradebook_lookup_session(proxy=self._proxy)
        lookup_session.get_gradebook(gradebook_id)  # to raise NotFound
        self._unassign_object_from_catalog(gradebook_column_id, gradebook_id)