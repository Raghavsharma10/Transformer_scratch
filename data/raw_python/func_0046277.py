def get_item_bank_session(self):
        """Gets the ``OsidSession`` associated with the item banking service.

        return: (osid.assessment.ItemBankSession) - an
                ``ItemBankSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_item_bank()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_item_bank()`` is ``true``.*

        """
        if not self.supports_item_bank():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ItemBankSession(runtime=self._runtime)