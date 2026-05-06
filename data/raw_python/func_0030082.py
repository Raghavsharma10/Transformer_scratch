def add_configuration(self, id, **kwargs):
        """
        Adds a configuration to the Specified Set
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.add_configuration(id, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param int id: Build Configuration Set id (required)
        :param BuildConfigurationRest body:
        :return: None
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.add_configuration_with_http_info(id, **kwargs)
        else:
            (data) = self.add_configuration_with_http_info(id, **kwargs)
            return data