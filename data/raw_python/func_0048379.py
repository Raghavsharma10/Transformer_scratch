def _get_agent_key(self, proxy=None):
        """Gets an agent key for session management.

        Side effect of setting a new proxy if one is sent along,
        and initializing the provider session map if agent key has
        not been seen before

        """
        if self._proxy is None:
            self._proxy = proxy
        if self._proxy is not None and self._proxy.has_effective_agent():
            agent_key = self._proxy.get_effective_agent_id()
        else:
            agent_key = None
        if agent_key not in self._provider_sessions:
            self._provider_sessions[agent_key] = dict()
        return agent_key