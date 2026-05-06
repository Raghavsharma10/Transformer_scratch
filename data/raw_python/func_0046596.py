def get_gradebook_column_gradebook_session(self):
        """Gets the session for retrieving gradebook column to gradebook mappings.

        return: (osid.grading.GradebookColumnGradebookSession) - a
                ``GradebookColumnGradebookSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented -
                ``supports_gradebook_column_gradebook()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_gradebook_column_gradebook()`` is ``true``.*

        """
        if not self.supports_gradebook_column_gradebook():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.GradebookColumnGradebookSession(runtime=self._runtime)