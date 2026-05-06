def remove_dependency(self, id, dependency_id, **kwargs):
        """
        Removes a configuration from the specified config set
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.remove_dependency(id, dependency_id, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param int id: Build configuration set id (required)
        :param int dependency_id: Build configuration id (required)
        :return: None
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.remove_dependency_with_http_info(id, dependency_id, **kwargs)
        else:
            (data) = self.remove_dependency_with_http_info(id, dependency_id, **kwargs)
            return data