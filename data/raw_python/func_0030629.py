def get_all_support_level(self, **kwargs):
        """
        Gets all Product Releases Support Level
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.get_all_support_level(callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :return: SupportLevelPage
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.get_all_support_level_with_http_info(**kwargs)
        else:
            (data) = self.get_all_support_level_with_http_info(**kwargs)
            return data