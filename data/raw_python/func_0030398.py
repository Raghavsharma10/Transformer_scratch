def clone(self, id, **kwargs):
        """
        Clones an existing Build Configuration
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.clone(id, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param int id: Build Configuration id (required)
        :return: BuildConfigurationSingleton
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.clone_with_http_info(id, **kwargs)
        else:
            (data) = self.clone_with_http_info(id, **kwargs)
            return data