def get_gradebook_column_admin_session(self):
        """Gets the ``OsidSession`` associated with the gradebook column administration service.

        return: (osid.grading.GradebookColumnAdminSession) - a
                ``GradebookColumnAdminSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_gradebook_column_admin()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_gradebook_column_admin()`` is ``true``.*

        """
        if not self.supports_gradebook_column_admin():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.GradebookColumnAdminSession(runtime=self._runtime)