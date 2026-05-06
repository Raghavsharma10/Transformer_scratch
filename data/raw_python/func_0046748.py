def assign_gradebook_column_to_gradebook(self, gradebook_column_id, gradebook_id):
        """Adds an existing ``GradebookColumn`` to a ``Gradebook``.

        arg:    gradebook_column_id (osid.id.Id): the ``Id`` of the
                ``GradebookColumn``
        arg:    gradebook_id (osid.id.Id): the ``Id`` of the
                ``Gradebook``
        raise:  AlreadyExists - ``gradebook_column_id`` is already
                assigned to ``gradebook_id``
        raise:  NotFound - ``gradebook_column_id`` or ``gradebook_id``
                not found
        raise:  NullArgument - ``gradebook_column_id`` or
                ``gradebook_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinAssignmentSession.assign_resource_to_bin
        mgr = self._get_provider_manager('GRADING', local=True)
        lookup_session = mgr.get_gradebook_lookup_session(proxy=self._proxy)
        lookup_session.get_gradebook(gradebook_id)  # to raise NotFound
        self._assign_object_to_catalog(gradebook_column_id, gradebook_id)