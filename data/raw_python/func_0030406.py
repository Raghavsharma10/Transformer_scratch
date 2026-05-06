def get_supported_generic_parameters(self, **kwargs):
        """
        Gets the minimal set of supported genericParameters and their description for the BuildConfiguration. There can be also other supported parameters not know by core.
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.get_supported_generic_parameters(callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :return: None
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.get_supported_generic_parameters_with_http_info(**kwargs)
        else:
            (data) = self.get_supported_generic_parameters_with_http_info(**kwargs)
            return data