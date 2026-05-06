def get_grade_system_lookup_session(self, proxy):
        """Gets the ``OsidSession`` associated with the grade system lookup service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.grading.GradeSystemLookupSession) - a
                ``GradeSystemLookupSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_grade_system_lookup()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_grade_system_lookup()`` is ``true``.*

        """
        if not self.supports_grade_system_lookup():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.GradeSystemLookupSession(proxy=proxy, runtime=self._runtime)