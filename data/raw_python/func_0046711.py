def get_gradebook_ids_by_grade_system(self, grade_system_id):
        """Gets the list of ``Gradebook``  ``Ids`` mapped to a ``GradeSystem``.

        arg:    grade_system_id (osid.id.Id): ``Id`` of a
                ``GradeSystem``
        return: (osid.id.IdList) - list of gradebook ``Ids``
        raise:  NotFound - ``grade_system_id`` is not found
        raise:  NullArgument - ``grade_system_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_bin_ids_by_resource
        mgr = self._get_provider_manager('GRADING', local=True)
        lookup_session = mgr.get_grade_system_lookup_session(proxy=self._proxy)
        lookup_session.use_federated_gradebook_view()
        grade_system = lookup_session.get_grade_system(grade_system_id)
        id_list = []
        for idstr in grade_system._my_map['assignedGradebookIds']:
            id_list.append(Id(idstr))
        return IdList(id_list)