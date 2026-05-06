def push_record_set(self, **kwargs):
        """
        Push build config set record to Brew.
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.push_record_set(callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param BuildConfigSetRecordPushRequestRest body:
        :return: list[ResultRest]
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.push_record_set_with_http_info(**kwargs)
        else:
            (data) = self.push_record_set_with_http_info(**kwargs)
            return data