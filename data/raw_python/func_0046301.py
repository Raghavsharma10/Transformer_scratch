def get_item_admin_session(self, proxy):
        """Gets the ``OsidSession`` associated with the item administration service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.assessment.ItemAdminSession) - an
                ``ItemAdminSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_item_admin()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_item_admin()`` is ``true``.*

        """
        if not self.supports_item_admin():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ItemAdminSession(proxy=proxy, runtime=self._runtime)