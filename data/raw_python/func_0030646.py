def get_dependency_graph(self, id, **kwargs):
        """
        Gets dependency graph for a Build Record (running or completed).
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.get_dependency_graph(id, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param int id: Build id. (required)
        :return: Singleton
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.get_dependency_graph_with_http_info(id, **kwargs)
        else:
            (data) = self.get_dependency_graph_with_http_info(id, **kwargs)
            return data