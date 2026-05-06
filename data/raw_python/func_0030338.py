def build(self, build_execution_configuration, **kwargs):
        """
        Triggers the build execution for a given configuration.
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.build(build_execution_configuration, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param str build_execution_configuration: Build Execution Configuration. See org.jboss.pnc.spi.executor.BuildExecutionConfiguration. (required)
        :param str username_triggered: Username who triggered the build. If empty current user is used.
        :param str callback_url: Optional Callback URL
        :return: None
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.build_with_http_info(build_execution_configuration, **kwargs)
        else:
            (data) = self.build_with_http_info(build_execution_configuration, **kwargs)
            return data