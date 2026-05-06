def get_effective_agent(self):
        """Gets the effective agent in use by this session.
        If is_authenticated() is true, then the effective agent may be
        the same as the agent returned by get_authenticated_agent(). If
        is_authenticated() is false, then the effective agent may be a
        default agent used for authorization by an unknwon or anonymous
        user.
        return: (osid.authentication.Agent) - the effective agent
        raise:  OperationFailed - unable to complete request
        compliance: mandatory - This method must be implemented.

        """
        if self._proxy is not None and self._proxy.has_authentication():
            return self._proxy.get_authentication().get_agent()
        elif self._proxy is not None and self._proxy.has_effective_agent():
            return Agent(identifier=self._proxy.get_effective_agent_id().get_identifier(),
                         namespace=self._proxy.get_effective_agent_id().get_namespace(),
                         authority=self._proxy.get_effective_agent_id().get_authority())
        else:
            return Agent(identifier='MC3GUE$T@MIT.EDU',
                         namespace='osid.agent.Agent',
                         authority='MIT-OEIT')