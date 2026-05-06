def get_effective_agent_id(self):
        """Gets the Id of the effective agent in use by this session.
        If is_authenticated() is true, then the effective agent may be
        the same as the agent returned by get_authenticated_agent(). If
        is_authenticated() is false, then the effective agent may be a
        default agent used for authorization by an unknwon or anonymous
        user.
        return: (osid.id.Id) - the effective agent
        compliance: mandatory - This method must be implemented.

        """
        if self.is_authenticated():
            return self._proxy.get_authentication().get_agent_id()
        elif self._proxy is not None and self._proxy.has_effective_agent():
            return self._proxy.get_effective_agent_id()
        else:
            return Id(identifier='MC3GUE$T@MIT.EDU',
                      namespace='osid.agent.Agent',
                      authority='MIT-OEIT')