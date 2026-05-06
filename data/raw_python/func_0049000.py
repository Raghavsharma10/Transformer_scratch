def get_proficiency_lookup_session(self, proxy):
        """Gets the ``OsidSession`` associated with the proficiency lookup service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.learning.ProficiencyLookupSession) - a
                ``ProficiencyLookupSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_proficiency_lookup()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_proficiency_lookup()`` is ``true``.*

        """
        if not self.supports_proficiency_lookup():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ProficiencyLookupSession(proxy=proxy, runtime=self._runtime)