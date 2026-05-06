def get_resource_agent_session(self):
        """Gets the session for retrieving resource agent mappings.

        return: (osid.resource.ResourceAgentSession) - a
                ``ResourceAgentSession``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_resource_agent()`` is
                ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_resource_agent()`` is ``true``.*

        """
        if not self.supports_resource_agent():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ResourceAgentSession(runtime=self._runtime)