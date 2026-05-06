def get_product_versions(self, id, **kwargs):
        """
        Get associated Product Versions of the specified Configuration
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.get_product_versions(id, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param int id: Build configuration id (required)
        :param int page_index: Page Index
        :param int page_size: Pagination size
        :param str sort: Sorting RSQL
        :param str q: RSQL Query
        :return: ProductVersionPage
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.get_product_versions_with_http_info(id, **kwargs)
        else:
            (data) = self.get_product_versions_with_http_info(id, **kwargs)
            return data