def get_build_configuration_audited(self, id, **kwargs):
        """
        Gets the audited build configuration for specific build record
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.get_build_configuration_audited(id, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param int id: BuildRecord id (required)
        :return: BuildConfigurationAuditedSingleton
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.get_build_configuration_audited_with_http_info(id, **kwargs)
        else:
            (data) = self.get_build_configuration_audited_with_http_info(id, **kwargs)
            return data