def add_distributed_artifact(self, id, **kwargs):
        """
        Adds an artifact to the list of distributed artifacts for this product milestone
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.add_distributed_artifact(id, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param int id: Product milestone id (required)
        :param ArtifactRest body:
        :return: None
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.add_distributed_artifact_with_http_info(id, **kwargs)
        else:
            (data) = self.add_distributed_artifact_with_http_info(id, **kwargs)
            return data