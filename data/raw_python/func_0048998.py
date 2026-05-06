def get_activity_objective_bank_session(self, proxy):
        """Gets the session for retrieving activity to objective bank mappings.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.learning.ActivityObjectiveBankSession) - an
                ``ActivityObjectiveBankSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_activity_objective_bank()``
                is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_activity_objective_bank()`` is ``true``.*

        """
        if not self.supports_activity_objective_bank():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ActivityObjectiveBankSession(proxy=proxy, runtime=self._runtime)