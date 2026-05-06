def get_all_by_product_id(self, product_id, **kwargs):
        """
        Gets all Build Configurations of a Product
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.get_all_by_product_id(product_id, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param int product_id: Product id (required)
        :param int page_index: Page Index
        :param int page_size: Pagination size
        :param str sort: Sorting RSQL
        :param str q: RSQL Query
        :return: BuildConfigurationPage
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.get_all_by_product_id_with_http_info(product_id, **kwargs)
        else:
            (data) = self.get_all_by_product_id_with_http_info(product_id, **kwargs)
            return data