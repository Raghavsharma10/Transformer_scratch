def get_all_for_project_0(self, project_id, **kwargs):
        """
        Gets the Build Records linked to a specific Project
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.get_all_for_project_0(project_id, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param int project_id: Project id (required)
        :param int page_index: Page index
        :param int page_size: Pagination size
        :param str sort: Sorting RSQL
        :param str q: RSQL query
        :return: BuildRecordPage
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.get_all_for_project_0_with_http_info(project_id, **kwargs)
        else:
            (data) = self.get_all_for_project_0_with_http_info(project_id, **kwargs)
            return data