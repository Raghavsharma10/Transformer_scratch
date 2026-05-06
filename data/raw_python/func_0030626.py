def get_ssh_credentials(self, id, **kwargs):
        """
        Gets ssh credentials for a build
        This GET request is for authenticated users only. The path for the endpoint is not restful to be able to authenticate this GET request only.
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.get_ssh_credentials(id, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param int id: BuildRecord id (required)
        :return: SshCredentialsSingleton
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.get_ssh_credentials_with_http_info(id, **kwargs)
        else:
            (data) = self.get_ssh_credentials_with_http_info(id, **kwargs)
            return data