def close_milestone(self, id, **kwargs):
        """
        Close/Release a Product Milestone
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.close_milestone(id, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param int id: Product Milestone id (required)
        :param ProductMilestoneRest body:
        :return: None
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.close_milestone_with_http_info(id, **kwargs)
        else:
            (data) = self.close_milestone_with_http_info(id, **kwargs)
            return data