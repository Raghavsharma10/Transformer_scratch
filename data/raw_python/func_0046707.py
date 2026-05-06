def get_grade_system_ids_by_gradebook(self, gradebook_id):
        """Gets the list of ``GradeSystem``  ``Ids`` associated with a ``Gradebook``.

        arg:    gradebook_id (osid.id.Id): ``Id`` of the ``Gradebook``
        return: (osid.id.IdList) - list of related grade system ``Ids``
        raise:  NotFound - ``gradebook_id`` is not found
        raise:  NullArgument - ``gradebook_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resource_ids_by_bin
        id_list = []
        for grade_system in self.get_grade_systems_by_gradebook(gradebook_id):
            id_list.append(grade_system.get_id())
        return IdList(id_list)