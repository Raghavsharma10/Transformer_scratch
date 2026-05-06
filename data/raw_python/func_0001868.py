def set_global(cls, user_agent=None, user_agent_config_yaml=None,
                   user_agent_lookup=None, **kwargs):
        # type: (Optional[str], Optional[str], Optional[str], Any) -> None
        """
        Set global user agent string

        Args:
            user_agent (Optional[str]): User agent string. HDXPythonLibrary/X.X.X- is prefixed.
            user_agent_config_yaml (Optional[str]): Path to YAML user agent configuration. Ignored if user_agent supplied. Defaults to ~/.useragent.yml.
            user_agent_lookup (Optional[str]): Lookup key for YAML. Ignored if user_agent supplied.

        Returns:
            None
        """
        cls.user_agent = cls._create(user_agent, user_agent_config_yaml, user_agent_lookup, **kwargs)