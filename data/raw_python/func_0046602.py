def get_grade_entry_lookup_session(self, proxy):
        """Gets the ``OsidSession`` associated with the grade entry lookup service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.grading.GradeEntryLookupSession) - a
                ``GradeEntryLookupSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_grade_entry_lookup()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_grade_entry_lookup()`` is ``true``.*

        """
        if not self.supports_grade_entry_lookup():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.GradeEntryLookupSession(proxy=proxy, runtime=self._runtime)