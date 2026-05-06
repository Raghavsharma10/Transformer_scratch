def start_r_creation_task_with_single_url(self, body, **kwargs):
        """
        Start Repository Creation task with url autodetect (internal vs. external).
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.start_r_creation_task_with_single_url(body, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param RepositoryCreationUrlAutoRest body: Task parameters. (required)
        :return: int
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.start_r_creation_task_with_single_url_with_http_info(body, **kwargs)
        else:
            (data) = self.start_r_creation_task_with_single_url_with_http_info(body, **kwargs)
            return data