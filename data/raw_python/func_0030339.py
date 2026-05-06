def build_task_completed(self, task_id, build_result, **kwargs):
        """
        Notifies the completion of externally managed build task process.
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.build_task_completed(task_id, build_result, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param int task_id: Build task id (required)
        :param str build_result: Build result (required)
        :return: None
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.build_task_completed_with_http_info(task_id, build_result, **kwargs)
        else:
            (data) = self.build_task_completed_with_http_info(task_id, build_result, **kwargs)
            return data