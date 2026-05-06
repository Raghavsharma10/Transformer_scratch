def get_hierarchy_lookup_session(self, proxy):
        """Gets the ``OsidSession`` associated with the hierarchy lookup service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.hierarchy.HierarchyLookupSession) - a
                ``HierarchyLookupSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_hierarchy_lookup()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_hierarchy_lookup()`` is ``true``.*

        """
        if not self.supports_hierarchy_lookup():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.HierarchyLookupSession(proxy=proxy, runtime=self._runtime)