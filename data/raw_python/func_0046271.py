def get_item_lookup_session(self):
        """Gets the ``OsidSession`` associated with the item lookup service.

        return: (osid.assessment.ItemLookupSession) - an
                ``ItemLookupSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_item_lookup()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_item_lookup()`` is ``true``.*

        """
        if not self.supports_item_lookup():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ItemLookupSession(runtime=self._runtime)