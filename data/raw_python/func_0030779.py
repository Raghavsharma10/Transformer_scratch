def status(self, build_record_id, **kwargs):
        """
        Latest push result of BuildRecord.
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.status(build_record_id, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param int build_record_id: Build Record id (required)
        :return: BuildRecordPushResultRest
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.status_with_http_info(build_record_id, **kwargs)
        else:
            (data) = self.status_with_http_info(build_record_id, **kwargs)
            return data