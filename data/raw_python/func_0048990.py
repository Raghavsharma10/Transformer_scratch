def get_objective_requisite_session(self, proxy):
        """Gets the session for examining objective requisites.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.learning.ObjectiveRequisiteSession) - an
                ``ObjectiveRequisiteSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_objective_requisite()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_objective_requisite()`` is ``true``.*

        """
        if not self.supports_objective_requisite():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ObjectiveRequisiteSession(proxy=proxy, runtime=self._runtime)