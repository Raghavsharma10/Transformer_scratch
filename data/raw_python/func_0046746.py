def get_gradebook_ids_by_gradebook_column(self, gradebook_column_id):
        """Gets the list of ``Gradebook``  ``Ids`` mapped to a ``GradebookColumn``.

        arg:    gradebook_column_id (osid.id.Id): ``Id`` of a
                ``GradebookColumn``
        return: (osid.id.IdList) - list of gradebook ``Ids``
        raise:  NotFound - ``gradebook_column_id`` is not found
        raise:  NullArgument - ``gradebook_column_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_bin_ids_by_resource
        mgr = self._get_provider_manager('GRADING', local=True)
        lookup_session = mgr.get_gradebook_column_lookup_session(proxy=self._proxy)
        lookup_session.use_federated_gradebook_view()
        gradebook_column = lookup_session.get_gradebook_column(gradebook_column_id)
        id_list = []
        for idstr in gradebook_column._my_map['assignedGradebookIds']:
            id_list.append(Id(idstr))
        return IdList(id_list)