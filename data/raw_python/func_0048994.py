def get_activity_lookup_session(self, proxy):
        """Gets the ``OsidSession`` associated with the activity lookup service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.learning.ActivityLookupSession) - an
                ``ActivityLookupSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_activity_lookup()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_activity_lookup()`` is ``true``.*

        """
        if not self.supports_activity_lookup():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ActivityLookupSession(proxy=proxy, runtime=self._runtime)