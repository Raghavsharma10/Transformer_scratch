def get_agent_lookup_session(self):
        """Gets the ``OsidSession`` associated with the agent lookup service.

        return: (osid.authentication.AgentLookupSession) - an
                ``AgentLookupSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_agent_lookup()`` is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_agent_lookup()`` is ``true``.*

        """
        if not self.supports_agent_lookup():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.AgentLookupSession(runtime=self._runtime)