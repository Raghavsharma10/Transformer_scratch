def get_gradebook_column_summary(self, gradebook_column_id):
        """Gets the ``GradebookColumnSummary`` for summary results.

        arg:    gradebook_column_id (osid.id.Id): ``Id`` of the
                ``GradebookColumn``
        return: (osid.grading.GradebookColumnSummary) - the gradebook
                column summary
        raise:  NotFound - ``gradebook_column_id`` is not found
        raise:  NullArgument - ``gradebook_column_id`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        raise:  Unimplemented - ``has_summary()`` is ``false``
        *compliance: mandatory -- This method is must be implemented.*

        """
        gradebook_column = self.get_gradebook_column(gradebook_column_id)
        summary_map = gradebook_column._my_map
        summary_map['gradebookColumnId'] = str(gradebook_column.ident)
        return GradebookColumnSummary(osid_object_map=summary_map,
                                      runtime=self._runtime,
                                      proxy=self._proxy)