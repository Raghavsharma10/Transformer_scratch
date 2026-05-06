def _create(cls, user_agent=None, user_agent_config_yaml=None,
                user_agent_lookup=None, **kwargs):
        # type: (Optional[str], Optional[str], Optional[str], Any) -> str
        """
        Get full user agent string

        Args:
            user_agent (Optional[str]): User agent string. HDXPythonLibrary/X.X.X- is prefixed.
            user_agent_config_yaml (Optional[str]): Path to YAML user agent configuration. Ignored if user_agent supplied. Defaults to ~/.useragent.yml.
            user_agent_lookup (Optional[str]): Lookup key for YAML. Ignored if user_agent supplied.

        Returns:
            str: Full user agent string

        """
        kwargs = UserAgent._environment_variables(**kwargs)
        if 'user_agent' in kwargs:
            user_agent = kwargs['user_agent']
            del kwargs['user_agent']
        prefix = kwargs.get('prefix')
        if prefix:
            del kwargs['prefix']
        else:
            prefix = 'HDXPythonUtilities/%s' % get_utils_version()
        if not user_agent:
            ua = cls._load(prefix, user_agent_config_yaml, user_agent_lookup)
        else:
            ua = cls._construct(kwargs, prefix, user_agent)
        return ua