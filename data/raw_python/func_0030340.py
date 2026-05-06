def cancel_bbuild(self, build_execution_configuration_id, **kwargs):
        """
        Cancel the build execution defined with given executionConfigurationId.
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.cancel_bbuild(build_execution_configuration_id, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param int build_execution_configuration_id: Build Execution Configuration ID. See org.jboss.pnc.spi.executor.BuildExecutionConfiguration. (required)
        :return: None
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.cancel_bbuild_with_http_info(build_execution_configuration_id, **kwargs)
        else:
            (data) = self.cancel_bbuild_with_http_info(build_execution_configuration_id, **kwargs)
            return data