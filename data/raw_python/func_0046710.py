def get_grade_systems_by_gradebooks(self, gradebook_ids):
        """Gets the list of grade systems corresponding to a list of ``Gradebooks``.

        arg:    gradebook_ids (osid.id.IdList): list of gradebook
                ``Ids``
        return: (osid.grading.GradeSystemList) - list of grade systems
        raise:  NullArgument - ``gradebook_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resources_by_bins
        grade_system_list = []
        for gradebook_id in gradebook_ids:
            grade_system_list += list(
                self.get_grade_systems_by_gradebook(gradebook_id))
        return objects.GradeSystemList(grade_system_list)