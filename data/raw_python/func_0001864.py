def _environment_variables(**kwargs):
        # type: (Any) -> Any
        """
        Overwrite keyword arguments with environment variables

        Args:
            **kwargs: See below
            user_agent (str): User agent string.

        Returns:
            kwargs: Changed keyword arguments

        """
        user_agent = os.getenv('USER_AGENT')
        if user_agent is not None:
            kwargs['user_agent'] = user_agent
        preprefix = os.getenv('PREPREFIX')
        if preprefix is not None:
            kwargs['preprefix'] = preprefix
        return kwargs