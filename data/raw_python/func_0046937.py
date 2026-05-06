def get_asset_admin_session(self, proxy):
        """Gets an asset administration session for creating, updating and deleting assets.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.repository.AssetAdminSession) - an
                ``AssetAdminSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_asset_admin()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_asset_admin()`` is ``true``.*

        """
        if not self.supports_asset_admin():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssetAdminSession(proxy=proxy, runtime=self._runtime)