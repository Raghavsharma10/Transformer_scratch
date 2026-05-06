def get_all_build_config_set_records(self, id, **kwargs):
        """
        Get all build config set execution records associated with this build config set, returns empty list if none are found
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.get_all_build_config_set_records(id, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param int id: Build config set id (required)
        :param int page_index: Page Index
        :param int page_size: Pagination size
        :param str sort: Sorting RSQL
        :param str q: RSQL Query
        :return: BuildConfigurationSetRecordPage
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.get_all_build_config_set_records_with_http_info(id, **kwargs)
        else:
            (data) = self.get_all_build_config_set_records_with_http_info(id, **kwargs)
            return data