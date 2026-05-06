def get_logged_user(self, **kwargs):
        """
        Gets logged user and in case not existing creates a new one
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.get_logged_user(callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :return: UserSingleton
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.get_logged_user_with_http_info(**kwargs)
        else:
            (data) = self.get_logged_user_with_http_info(**kwargs)
            return data