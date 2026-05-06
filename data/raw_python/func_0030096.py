def get_all_for_bc_set_record(self, id, **kwargs):
        """
        Gets running Build Records for a specific Build Configuration Set Record.
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.get_all_for_bc_set_record(id, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param int id: Build Configuration Set id (required)
        :param int page_index: Page Index
        :param int page_size: Pagination size
        :param str search: Since this endpoint does not support queries, fulltext search is hard-coded for some predefined fields (record id, configuration name) and performed using this argument. Empty string leaves all data unfiltered.
        :return: BuildRecordPage
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.get_all_for_bc_set_record_with_http_info(id, **kwargs)
        else:
            (data) = self.get_all_for_bc_set_record_with_http_info(id, **kwargs)
            return data