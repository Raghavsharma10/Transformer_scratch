def get_agents_by_search(self, agent_query, agent_search):
        """Pass through to provider AgentSearchSession.get_agents_by_search"""
        # Implemented from azosid template for -
        # osid.resource.ResourceSearchSession.get_resources_by_search_template
        if not self._can('search'):
            raise PermissionDenied()
        return self._provider_session.get_agents_by_search(agent_query, agent_search)