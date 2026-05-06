def get_repour_logs(self, id, **kwargs):
        """
        Gets repour logs for specific Build Record
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.get_repour_logs(id, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param int id: BuildRecord id (required)
        :return: str
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.get_repour_logs_with_http_info(id, **kwargs)
        else:
            (data) = self.get_repour_logs_with_http_info(id, **kwargs)
            return data