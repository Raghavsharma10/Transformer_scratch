def notify_task(self, task_id, **kwargs):
        """
        Notify PNC about a BPM task event. Accepts polymorphic JSON {\"eventType\": \"string\"} based on \"eventType\" field.
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.notify_task(task_id, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param int task_id: BPM task ID (required)
        :return: None
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.notify_task_with_http_info(task_id, **kwargs)
        else:
            (data) = self.notify_task_with_http_info(task_id, **kwargs)
            return data