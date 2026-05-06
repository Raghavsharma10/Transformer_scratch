def get_activity_objective_bank_assignment_session(self, proxy):
        """Gets the session for assigning activity to objective bank mappings.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.learning.ActivityObjectiveBankAssignmentSession) -
                an ``ActivityObjectiveBankAssignmentSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented -
                ``supports_activity_objective_bank_assignment()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_activity_objective_bank_assignment()`` is ``true``.*

        """
        if not self.supports_activity_objective_bank_assignment():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ActivityObjectiveBankAssignmentSession(proxy=proxy, runtime=self._runtime)