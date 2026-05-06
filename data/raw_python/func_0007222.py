def from_env(cls, default_timeout=DEFAULT_TIMEOUT_SECONDS):
        """Return a client configured from environment variables.

        Essentially copying this:
        https://github.com/docker/docker-py/blob/master/docker/client.py#L43.

        The environment variables looked for are the following:

        .. envvar:: SALTANT_API_URL

            The URL of the saltant API. For example,
            https://shahlabjobs.ca/api/.

        .. envvar:: SALTANT_AUTH_TOKEN

            The registered saltant user's authentication token.

        Example:

            >>> from saltant.client import from_env
            >>> client = from_env()

        Args:
            default_timeout (int, optional): The maximum number of
                seconds to wait for a request to complete. Defaults to
                90 seconds.

        Returns:
            :class:`Client`: A saltant API client object.

        Raises:
            :class:`saltant.exceptions.BadEnvironmentError`: The user
                has an incorrectly configured environment.
        """
        # Get variables from environment
        try:
            base_api_url = os.environ["SALTANT_API_URL"]
        except KeyError:
            raise BadEnvironmentError("SALTANT_API_URL not defined!")

        try:
            # Try to get an auth token
            auth_token = os.environ["SALTANT_AUTH_TOKEN"]
        except KeyError:
            raise BadEnvironmentError("SALTANT_AUTH_TOKEN not defined!")

        # Return the configured client
        return cls(
            base_api_url=base_api_url,
            auth_token=auth_token,
            default_timeout=default_timeout,
        )