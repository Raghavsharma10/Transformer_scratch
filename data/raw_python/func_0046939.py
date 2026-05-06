def get_asset_composition_design_session(self, proxy):
        """Gets the session for creating asset compositions.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.repository.AssetCompositionDesignSession) - an
                ``AssetCompositionDesignSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_asset_composition_design()``
                is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_asset_composition_design()`` is ``true``.*

        """
        if not self.supports_asset_composition_design():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AssetCompositionDesignSession(proxy=proxy, runtime=self._runtime)