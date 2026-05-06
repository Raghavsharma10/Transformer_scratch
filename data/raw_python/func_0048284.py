def get_resource_agent_assignment_session(self, proxy):
        """Gets the session for assigning agents to resources.

        arg:    proxy (osid.proxy.Proxy): a proxy
        return: (osid.resource.ResourceAgentAssignmentSession) - a
                ``ResourceAgentAssignmentSession``
        raise:  NullArgument - ``proxy`` is ``null``
        raise:  OperationFailed - unable to complete request
        raise:  Unimplemented - ``supports_resource_agent_assignment()``
                is ``false``
        *compliance: optional -- This method must be implemented if
        ``supports_resource_agent_assignment()`` is ``true``.*

        """
        if not self.supports_resource_agent_assignment():
            raise errors.Unimplemented()
        # pylint: disable=no-member
        return sessions.ResourceAgentAssignmentSession(proxy=proxy, runtime=self._runtime)