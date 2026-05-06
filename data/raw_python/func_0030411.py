def update_and_get_audited(self, id, **kwargs):
        """
        Updates an existing Build Configuration and returns BuildConfigurationAudited entity
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.update_and_get_audited(id, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param int id: Build Configuration id (required)
        :param BuildConfigurationRest body:
        :return: BuildConfigurationAuditedSingleton
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.update_and_get_audited_with_http_info(id, **kwargs)
        else:
            (data) = self.update_and_get_audited_with_http_info(id, **kwargs)
            return data