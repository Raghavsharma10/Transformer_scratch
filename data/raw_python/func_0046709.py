def get_grade_system_ids_by_gradebooks(self, gradebook_ids):
        """Gets the list of ``GradeSystem Ids`` corresponding to a list of ``Gradebooks``.

        arg:    gradebook_ids (osid.id.IdList): list of gradebook
                ``Ids``
        return: (osid.id.IdList) - list of grade systems ``Ids``
        raise:  NullArgument - ``gradebook_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resource_ids_by_bins
        id_list = []
        for grade_system in self.get_grade_systems_by_gradebooks(gradebook_ids):
            id_list.append(grade_system.get_id())
        return IdList(id_list)