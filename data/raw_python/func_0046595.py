def get_gradebook_column_admin_session_for_gradebook(self, gradebook_id):
        """Gets the ``OsidSession`` associated with the gradebook column admin service for the given gradebook.

        arg:    gradebook_id (osid.id.Id): the ``Id`` of the gradebook
        return: (osid.grading.GradebookColumnAdminSession) - ``a
                GradebookColumnAdminSession``
        raise:  NotFound - ``gradebook_id`` not found
        raise:  NullArgument - ``gradebook_id`` is ``null``
        raise:  OperationFailed - ``unable to complete request``
        raise:  Unimplemented - ``supports_gradebook_column_admin()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_gradebook_column_admin()`` and
        ``supports_visible_federation()`` are ``true``.*

        """
        if not self.supports_gradebook_column_admin():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.GradebookColumnAdminSession(gradebook_id, runtime=self._runtime)