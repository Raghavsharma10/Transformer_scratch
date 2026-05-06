def get_revision(self, id, rev, **kwargs):
        """
        Get specific audited revision of this build configuration
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.get_revision(id, rev, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param int id: Build configuration id (required)
        :param int rev: Build configuration rev (required)
        :return: BuildConfigurationAuditedSingleton
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.get_revision_with_http_info(id, rev, **kwargs)
        else:
            (data) = self.get_revision_with_http_info(id, rev, **kwargs)
            return data