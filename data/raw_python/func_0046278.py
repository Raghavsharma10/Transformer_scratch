def get_item_bank_assignment_session(self):
        """Gets the ``OsidSession`` associated with the item bank assignment service.

        return: (osid.assessment.ItemBankAssignmentSession) - an
                ``ItemBankAssignmentSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_item_bank_assignment()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_item_bank_assignment()`` is ``true``.*

        """
        if not self.supports_item_bank_assignment():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ItemBankAssignmentSession(runtime=self._runtime)