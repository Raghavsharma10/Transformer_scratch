def get_objective_requisite_assignment_session(self, proxy):
        """Gets the session for managing objective requisites.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.learning.ObjectiveRequisiteAssignmentSession) - an
                ``ObjectiveRequisiteAssignmentSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented -
                ``supports_objective_requisite_assignment()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_objective_requisite_assignment()`` is ``true``.*

        """
        if not self.supports_objective_requisite_assignment():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ObjectiveRequisiteAssignmentSession(proxy=proxy, runtime=self._runtime)