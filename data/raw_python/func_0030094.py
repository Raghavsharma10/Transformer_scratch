def cancel_all_builds_in_group(self, id, **kwargs):
        """
        Cancel all builds running in the build group
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.cancel_all_builds_in_group(id, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param int id: Build Configuration Set id (required)
        :return: None
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.cancel_all_builds_in_group_with_http_info(id, **kwargs)
        else:
            (data) = self.cancel_all_builds_in_group_with_http_info(id, **kwargs)
            return data