def get(cls, user_agent=None, user_agent_config_yaml=None, user_agent_lookup=None, **kwargs):
        # type: (Optional[str], Optional[str], Optional[str], Any) -> str
        """
        Get full user agent string from parameters if supplied falling back on global user agent if set.

        Args:
            user_agent (Optional[str]): User agent string. HDXPythonLibrary/X.X.X- is prefixed.
            user_agent_config_yaml (Optional[str]): Path to YAML user agent configuration. Ignored if user_agent supplied. Defaults to ~/.useragent.yml.
            user_agent_lookup (Optional[str]): Lookup key for YAML. Ignored if user_agent supplied.

        Returns:
            str: Full user agent string

        """
        if user_agent or user_agent_config_yaml or 'user_agent' in UserAgent._environment_variables(**kwargs):
            return UserAgent._create(user_agent, user_agent_config_yaml, user_agent_lookup, **kwargs)
        if cls.user_agent:
            return cls.user_agent
        else:
            raise UserAgentError(
                'You must either set the global user agent: UserAgent.set_global(...) or pass in user agent parameters!')