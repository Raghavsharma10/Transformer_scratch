def get_gradebook_columns_by_gradebook(self, gradebook_id):
        """Gets the list of gradebook columns associated with a ``Gradebook``.

        arg:    gradebook_id (osid.id.Id): ``Id`` of the ``Gradebook``
        return: (osid.grading.GradebookColumnList) - list of related
                gradebook columns
        raise:  NotFound - ``gradebook_id`` is not found
        raise:  NullArgument - ``gradebook_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resources_by_bin
        mgr = self._get_provider_manager('GRADING', local=True)
        lookup_session = mgr.get_gradebook_column_lookup_session_for_gradebook(gradebook_id, proxy=self._proxy)
        lookup_session.use_isolated_gradebook_view()
        return lookup_session.get_gradebook_columns()