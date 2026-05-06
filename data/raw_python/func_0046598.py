def get_grade_system_lookup_session_for_gradebook(self, gradebook_id, proxy):
        """Gets the ``OsidSession`` associated with the grade system lookup service for the given gradebook.

        arg:    gradebook_id (osid.id.Id): the ``Id`` of the gradebook
        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.grading.GradeSystemLookupSession) - ``a
                GradeSystemLookupSession``
        raise:  NotFound - ``gradebook_id`` not found
        raise:  NullArgument - ``gradebook_id`` or ``proxy`` is ``null``
        raise:  OperationFailed - ``unable to complete request``
        raise:  Unimplemented - ``supports_grade_system_lookup()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_grade_system_lookup()`` and
        ``supports_visible_federation()`` are ``true``.*

        """
        if not self.supports_grade_system_lookup():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.GradeSystemLookupSession(gradebook_id, proxy, self._runtime)