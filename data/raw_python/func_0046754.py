def get_root_gradebooks(self):
        """Gets the root gradebooks in this gradebook hierarchy.

        return: (osid.grading.GradebookList) - the root gradebooks
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method is must be implemented.*

        """
        # Implemented from template for
        # osid.resource.BinHierarchySession.get_root_bins
        if self._catalog_session is not None:
            return self._catalog_session.get_root_catalogs()
        return GradebookLookupSession(
            self._proxy,
            self._runtime).get_gradebooks_by_ids(list(self.get_root_gradebook_ids()))