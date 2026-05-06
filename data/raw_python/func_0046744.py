def get_gradebook_column_ids_by_gradebooks(self, gradebook_ids):
        """Gets the list of ``GradebookColumn Ids`` corresponding to a list of ``Gradebooks``.

        arg:    gradebook_ids (osid.id.IdList): list of gradebook
                ``Ids``
        return: (osid.id.IdList) - list of gradebook column ``Ids``
        raise:  NullArgument - ``gradebook_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resource_ids_by_bins
        id_list = []
        for gradebook_column in self.get_gradebook_columns_by_gradebooks(gradebook_ids):
            id_list.append(gradebook_column.get_id())
        return IdList(id_list)