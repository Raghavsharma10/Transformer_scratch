def get_family_lookup_session(self, proxy=None, *args, **kwargs):
        """Gets the ``OsidSession`` associated with the family lookup service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.relationship.FamilyLookupSession) - a
                ``FamilyLookupSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_family_lookup()`` is
                ``false``
        *compliance: optional -- This method must be implemented if ``supports_family_lookup()`` is ``true``.*

        """
        if not self.supports_family_lookup():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        proxy = self._convert_proxy(proxy)
        try:
            session = sessions.FamilyLookupSession(proxy=proxy, runtime=self._runtime, **kwargs)
        except AttributeError:
            raise OperationFailed()
        return session