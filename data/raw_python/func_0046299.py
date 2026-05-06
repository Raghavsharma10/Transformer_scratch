def get_item_search_session(self, proxy):
        """Gets the ``OsidSession`` associated with the item search service.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.assessment.ItemSearchSession) - an
                ``ItemSearchSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_item_search()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_item_search()`` is ``true``.*

        """
        if not self.supports_item_search():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ItemSearchSession(proxy=proxy, runtime=self._runtime)