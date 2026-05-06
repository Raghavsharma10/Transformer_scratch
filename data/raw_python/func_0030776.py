def push(self, **kwargs):
        """
        Push build record results to Brew.
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.push(callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param BuildRecordPushRequestRest body:
        :return: list[ResultRest]
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.push_with_http_info(**kwargs)
        else:
            (data) = self.push_with_http_info(**kwargs)
            return data