def _load(cls, prefix, user_agent_config_yaml, user_agent_lookup=None):
        # type: (str, str, Optional[str]) -> str
        """
        Load user agent YAML file

        Args:
            prefix (str): Text to put at start of user agent
            user_agent_config_yaml (str): Path to user agent YAML file
            user_agent_lookup (Optional[str]): Lookup key for YAML. Ignored if user_agent supplied.

        Returns:
            str: user agent

        """
        if not user_agent_config_yaml:
            user_agent_config_yaml = cls.default_user_agent_config_yaml
            logger.info(
                'No user agent or user agent config file given. Using default user agent config file: %s.' % user_agent_config_yaml)
        if not isfile(user_agent_config_yaml):
            raise UserAgentError(
                "User_agent should be supplied in a YAML config file. It can be your project's name for example.")
        logger.info('Loading user agent config from: %s' % user_agent_config_yaml)
        user_agent_config_dict = load_yaml(user_agent_config_yaml)
        if user_agent_lookup:
            user_agent_config_dict = user_agent_config_dict.get(user_agent_lookup)
        if not user_agent_config_dict:
            raise UserAgentError("No user agent information read from: %s" % user_agent_config_yaml)
        ua = user_agent_config_dict.get('user_agent')
        return cls._construct(user_agent_config_dict, prefix, ua)