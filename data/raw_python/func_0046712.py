def get_gradebooks_by_grade_system(self, grade_system_id):
        """Gets the list of ``Gradebooks`` mapped to a ``GradeSystem``.

        arg:    grade_system_id (osid.id.Id): ``Id`` of a
                ``GradeSystem``
        return: (osid.grading.GradebookList) - list of gradebooks
        raise:  NotFound - ``grade_system_id`` is not found
        raise:  NullArgument - ``grade_system_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_bins_by_resource
        mgr = self._get_provider_manager('GRADING', local=True)
        lookup_session = mgr.get_gradebook_lookup_session(proxy=self._proxy)
        return lookup_session.get_gradebooks_by_ids(
            self.get_gradebook_ids_by_grade_system(grade_system_id))