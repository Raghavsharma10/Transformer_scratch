def update_build_configuration_sets(self, id, **kwargs):
        """
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.update_build_configuration_sets(id, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param int id: Product Version id (required)
        :param list[BuildConfigurationSetRest] body:
        :return: None
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.update_build_configuration_sets_with_http_info(id, **kwargs)
        else:
            (data) = self.update_build_configuration_sets_with_http_info(id, **kwargs)
            return data