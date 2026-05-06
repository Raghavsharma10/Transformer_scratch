def get_objective_sequencing_session(self, proxy):
        """Gets the session for sequencing objectives.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.learning.ObjectiveSequencingSession) - an
                ``ObjectiveSequencingSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_objective_sequencing()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_objective_sequencing()`` is ``true``.*

        """
        if not self.supports_objective_sequencing():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ObjectiveSequencingSession(proxy=proxy, runtime=self._runtime)