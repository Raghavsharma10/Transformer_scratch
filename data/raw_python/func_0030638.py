def get_all_for_build_configuration(self, configuration_id, **kwargs):
        """
        Gets the Build Records linked to a specific Build Configuration
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.get_all_for_build_configuration(configuration_id, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param int configuration_id: Build Configuration id (required)
        :param int page_index: Page Index
        :param int page_size: Pagination size
        :param str sort: Sorting RSQL
        :param str q: RSQL Query
        :return: BuildRecordPage
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.get_all_for_build_configuration_with_http_info(configuration_id, **kwargs)
        else:
            (data) = self.get_all_for_build_configuration_with_http_info(configuration_id, **kwargs)
            return data