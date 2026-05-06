def get_distributed_builds(self, id, **kwargs):
        """
        Gets the set of builds which produced artifacts distributed/shipped in a Product Milestone
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.get_distributed_builds(id, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param int id: Product Milestone id (required)
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
            return self.get_distributed_builds_with_http_info(id, **kwargs)
        else:
            (data) = self.get_distributed_builds_with_http_info(id, **kwargs)
            return data