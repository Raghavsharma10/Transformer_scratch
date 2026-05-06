def get_gradebooks_by_gradebook_column(self, gradebook_column_id):
        """Gets the list of ``Gradebooks`` mapped to a ``GradebookColumn``.

        arg:    gradebook_column_id (osid.id.Id): ``Id`` of a
                ``GradebookColumn``
        return: (osid.grading.GradebookList) - list of gradebooks
        raise:  NotFound - ``gradebook_column_id`` is not found
        raise:  NullArgument - ``gradebook_column_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_bins_by_resource
        mgr = self._get_provider_manager('GRADING', local=True)
        lookup_session = mgr.get_gradebook_lookup_session(proxy=self._proxy)
        return lookup_session.get_gradebooks_by_ids(
            self.get_gradebook_ids_by_gradebook_column(gradebook_column_id))