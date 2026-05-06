def get_item_query_session(self):
        """Gets the ``OsidSession`` associated with the item query service.

        return: (osid.assessment.ItemQuerySession) - an
                ``ItemQuerySession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_item_query()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_item_query()`` is ``true``.*

        """
        if not self.supports_item_query():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ItemQuerySession(runtime=self._runtime)