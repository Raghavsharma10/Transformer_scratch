def get_child_gradebooks(self, gradebook_id):
        """Gets the children of the given gradebook.

        arg:    gradebook_id (osid.id.Id): the ``Id`` to query
        return: (osid.grading.GradebookList) - the children of the
                gradebook
        raise:  NotFound - ``gradebook_id`` is not found
        raise:  NullArgument - ``gradebook_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.get_child_bins
        if self._catalog_session is not None:
            return self._catalog_session.get_child_catalogs(catalog_id=gradebook_id)
        return GradebookLookupSession(
            self._proxy,
            self._runtime).get_gradebooks_by_ids(
                list(self.get_child_gradebook_ids(gradebook_id)))