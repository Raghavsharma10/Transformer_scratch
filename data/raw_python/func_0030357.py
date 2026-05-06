def get_bpm_task_by_id(self, task_id, **kwargs):
        """
        Get single (recently) active BPM task.
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.get_bpm_task_by_id(task_id, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param int task_id: BPM task ID (required)
        :return: BpmTaskRestSingleton
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.get_bpm_task_by_id_with_http_info(task_id, **kwargs)
        else:
            (data) = self.get_bpm_task_by_id_with_http_info(task_id, **kwargs)
            return data