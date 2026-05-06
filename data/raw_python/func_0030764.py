def match(self, search, **kwargs):
        """
        Searches for Repository Configurations based on internal or external url, ignoring the protocol and \".git\" suffix. Only exact matches are returned.
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.match(search, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param str search: Url to search for (required)
        :param int page_index: Page Index
        :param int page_size: Pagination size
        :param str sort: Sorting RSQL
        :return: RepositoryConfigurationPage
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.match_with_http_info(search, **kwargs)
        else:
            (data) = self.match_with_http_info(search, **kwargs)
            return data