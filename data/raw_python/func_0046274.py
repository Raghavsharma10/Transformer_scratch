def get_item_query_session_for_bank(self, bank_id):
        """Gets the ``OsidSession`` associated with the item query service for the given bank.

        arg:    bank_id (osid.id.Id): the ``Id`` of the bank
        return: (osid.assessment.ItemQuerySession) - ``an
                _item_query_session``
        raise:  NotFound - ``bank_id`` not found
        raise:  NullArgument - ``bank_id`` is ``null``
        raise:  OperationFailed - ``unable to complete request``
        raise:  Unimplemented - ``supports_item_query()`` or
                ``supports_visible_federation()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_item_query()`` and ``supports_visible_federation()``
        are ``true``.*

        """
        if not self.supports_item_query():
            raise errors.Unimplemented()
        ##
        # Also include check to see if the catalog Id is found otherwise raise errors.NotFound
        ##
        # pylint: disable=no-member
        return sessions.ItemQuerySession(bank_id, runtime=self._runtime)