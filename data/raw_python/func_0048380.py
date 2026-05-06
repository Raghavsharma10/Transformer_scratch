def _get_provider_sessions(self):
        """Gets list of all known provider sessions."""
        agent_key = self._get_agent_key()
        sessions = []
        for session_name in self._provider_sessions[agent_key]:
            sessions.append(self._provider_sessions[agent_key][session_name])
        return sessions