def get_gradebook_columns_by_gradebooks(self, gradebook_ids):
        """Gets the list of gradebook columns corresponding to a list of ``Gradebooks``.

        arg:    gradebook_ids (osid.id.IdList): list of gradebook
                ``Ids``
        return: (osid.grading.GradebookColumnList) - list of gradebook
                columns
        raise:  NullArgument - ``gradebook_ids`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.resource.ResourceBinSession.get_resources_by_bins
        gradebook_column_list = []
        for gradebook_id in gradebook_ids:
            gradebook_column_list += list(
                self.get_gradebook_columns_by_gradebook(gradebook_id))
        return objects.GradebookColumnList(gradebook_column_list)