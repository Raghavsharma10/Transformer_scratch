def get_bank_query_session(self):
        """Gets the OsidSession associated with the bank query service.

        return: (osid.assessment.BankQuerySession) - a
                ``BankQuerySession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_bank_query() is false``
        *compliance: optional -- This method must be implemented if
        ``supports_bank_query()`` is true.*

        """
        if not self.supports_bank_query():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.BankQuerySession(runtime=self._runtime)