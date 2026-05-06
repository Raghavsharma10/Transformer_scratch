def get_bank_lookup_session(self):
        """Gets the OsidSession associated with the bank lookup service.

        return: (osid.assessment.BankLookupSession) - a
                ``BankLookupSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_bank_lookup() is false``
        *compliance: optional -- This method must be implemented if
        ``supports_bank_lookup()`` is true.*

        """
        if not self.supports_bank_lookup():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.BankLookupSession(runtime=self._runtime)