def create_new(self, **kwargs):
        """
        Creates a new License
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.create_new(callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param LicenseRest body:
        :return: LicenseSingleton
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.create_new_with_http_info(**kwargs)
        else:
            (data) = self.create_new_with_http_info(**kwargs)
            return data