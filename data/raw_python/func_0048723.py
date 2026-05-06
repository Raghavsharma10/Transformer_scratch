def _can(self, func_name, qualifier_id=None):
        """Tests if the named function is authorized with agent and qualifier.

        Also, caches authz's in a dict.  It is expected that this will not grow to big, as
        there are typically only a small number of qualifier + function combinations to
        store for the agent.  However, if this becomes an issue, we can switch to something
        like cachetools.

        """
        function_id = self._get_function_id(func_name)
        if qualifier_id is None:
            qualifier_id = self._qualifier_id
        agent_id = self.get_effective_agent_id()
        try:
            return self._authz_cache[str(agent_id) + str(function_id) + str(qualifier_id)]
        except KeyError:
            authz = self._authz_session.is_authorized(agent_id=agent_id,
                                                      function_id=function_id,
                                                      qualifier_id=qualifier_id)
            self._authz_cache[str(agent_id) + str(function_id) + str(qualifier_id)] = authz
            return authz