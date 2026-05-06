def get_grade_entries_for_gradebook_column_on_date(self, gradebook_column_id, from_, to):
        """Gets a ``GradeEntryList`` for the given gradebook column and effective during the entire given date range inclusive but not confined to the date range.

        arg:    gradebook_column_id (osid.id.Id): a gradebook column
                ``Id``
        arg:    from (osid.calendaring.DateTime): start of date range
        arg:    to (osid.calendaring.DateTime): end of date range
        return: (osid.grading.GradeEntryList) - the returned
                ``GradeEntry`` list
        raise:  InvalidArgument - ``from`` is greater than ``to``
        raise:  NullArgument - ``gradebook_column_id, from, or to`` is
                ``null``
        raise:  OperationFailed - unable to complete request
        raise:  PermissionDenied - authorization failure
        *compliance: mandatory -- This method must be implemented.*

        """
        # Implemented from template for
        # osid.relationship.RelationshipLookupSession.get_relationships_for_source_on_date
        grade_entry_list = []
        for grade_entry in self.get_grade_entries_for_gradebook_column(gradebook_column_id):
            if overlap(from_, to, grade_entry.start_date, grade_entry.end_date):
                grade_entry_list.append(grade_entry)
        return objects.GradeEntryList(grade_entry_list, runtime=self._runtime)