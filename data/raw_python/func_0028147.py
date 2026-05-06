def call_remoteckan(self, *args, **kwargs):
        # type: (Any, Any) -> Dict
        """
        Calls the remote CKAN

        Args:
            *args: Arguments to pass to remote CKAN call_action method
            **kwargs: Keyword arguments to pass to remote CKAN call_action method

        Returns:
            Dict: The response from the remote CKAN call_action method

        """
        requests_kwargs = kwargs.get('requests_kwargs', dict())
        credentials = self._get_credentials()
        if credentials:
            requests_kwargs['auth'] = credentials
        kwargs['requests_kwargs'] = requests_kwargs
        apikey = kwargs.get('apikey', self.get_api_key())
        kwargs['apikey'] = apikey
        return self.remoteckan().call_action(*args, **kwargs)