def get_item_search_session_for_bank(self, bank_id, proxy):
        """Gets the ``OsidSession`` associated with the item search service for the given bank.

        arg:    bank_id (osid.id.Id): the ``Id`` of the bank
        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.assessment.ItemSearchSession) - ``an
                _item_search_session``
        raise:  NotFound - ``bank_id`` not found
        raise:  NullArgument - ``bank_id`` or ``porxy`` is ``null``
        raise:  OperationFailed - ``unable to complete request``
        raise:  Unimplemented - ``supports_item_search()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_item_search()`` and ``supports_visible_federation()``
        are ``true``.*

        """
        if not self.supports_item_search():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.ItemSearchSession(bank_id, proxy, self._runtime)