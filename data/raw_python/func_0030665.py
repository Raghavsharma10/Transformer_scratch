def remove_distributed_artifact(self, id, artifact_id, **kwargs):
        """
        Removes an artifact from the specified product milestone
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.remove_distributed_artifact(id, artifact_id, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param int id: Product milestone id (required)
        :param int artifact_id: Artifact id (required)
        :return: None
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.remove_distributed_artifact_with_http_info(id, artifact_id, **kwargs)
        else:
            (data) = self.remove_distributed_artifact_with_http_info(id, artifact_id, **kwargs)
            return data