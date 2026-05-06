def get_grade_entry_query_session_for_gradebook(self, gradebook_id):
        """Gets the ``OsidSession`` associated with the grade entry query service for the given gradebook.

        arg:    gradebook_id (osid.id.Id): the ``Id`` of the gradebook
        return: (osid.grading.GradeEntryQuerySession) - ``a
                GradeEntryQuerySession``
        raise:  NotFound - ``gradebook_id`` not found
        raise:  NullArgument - ``gradebook_id`` is ``null``
        raise:  OperationFailed - ``unable to complete request``
        raise:  Unimplemented - ``supports_grade_entry_query()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_grade_entry_query()`` and
        ``supports_visible_federation()`` are ``true``.*

        """
        if not self.supports_grade_entry_query():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.GradeEntryQuerySession(gradebook_id, runtime=self._runtime)