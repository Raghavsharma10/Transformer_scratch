def _construct(configdict, prefix, ua):
        # type: (Dict, str, str) -> str
        """
        Construct user agent

        Args:
            configdict (str): Additional configuration for user agent
            prefix (str): Text to put at start of user agent
            ua (str): Custom user agent text

        Returns:
            str: Full user agent string

        """
        if not ua:
            raise UserAgentError("User_agent parameter missing. It can be your project's name for example.")
        preprefix = configdict.get('preprefix')
        if preprefix:
            user_agent = '%s:' % preprefix
        else:
            user_agent = ''
        if prefix:
            user_agent = '%s%s-' % (user_agent, prefix)
        user_agent = '%s%s' % (user_agent, ua)
        return user_agent