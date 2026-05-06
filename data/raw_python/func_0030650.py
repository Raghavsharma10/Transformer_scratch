def query_by_attribute(self, key, value, **kwargs):
        """
        Get Build Records by attribute.
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.query_by_attribute(key, value, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param str key: Attribute key (required)
        :param str value: Attribute value (required)
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
            return self.query_by_attribute_with_http_info(key, value, **kwargs)
        else:
            (data) = self.query_by_attribute_with_http_info(key, value, **kwargs)
            return data