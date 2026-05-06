def get_objective_bank_lookup_session(self, *args, **kwargs):
        """Gets the OsidSession associated with the objective bank lookup
        service.

        return: (osid.learning.ObjectiveBankLookupSession) - an
                ObjectiveBankLookupSession
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - supports_objective_bank_lookup() is
                false
        compliance: optional - This method must be implemented if
                    supports_objective_bank_lookup() is true.

        """
        if not self.supports_objective_bank_lookup():
            raise Unimplemented()
        try:
            from . import sessions
        except ImportError:
            raise OperationFailed()
        try:
            session = sessions.ObjectiveBankLookupSession(runtime=self._runtime)
        except AttributeError:
            raise OperationFailed()
        return session