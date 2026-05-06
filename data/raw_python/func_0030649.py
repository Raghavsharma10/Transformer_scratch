def put_attribute(self, id, key, value, **kwargs):
        """
        Add attribute to the BuildRecord.
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.put_attribute(id, key, value, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param int id: BuildRecord id (required)
        :param str key: Attribute key (required)
        :param str value: Attribute value (required)
        :return: None
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.put_attribute_with_http_info(id, key, value, **kwargs)
        else:
            (data) = self.put_attribute_with_http_info(id, key, value, **kwargs)
            return data