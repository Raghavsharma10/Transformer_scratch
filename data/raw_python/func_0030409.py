def trigger(self, id, **kwargs):
        """
        Triggers a build of a specific Build Configuration
        
        This method makes a synchronous HTTP request by default. To make an
        asynchronous HTTP request, please define a `callback` function
        to be invoked when receiving the response.
        >>> def callback_function(response):
        >>>     pprint(response)
        >>>
        >>> thread = api.trigger(id, callback=callback_function)

        :param callback function: The callback function
            for asynchronous request. (optional)
        :param int id: Build Configuration id (required)
        :param str callback_url: Optional Callback URL
        :param bool temporary_build: Is it a temporary build or a standard build?
        :param bool force_rebuild: DEPRECATED: Use RebuildMode.
        :param bool build_dependencies: Should we build also dependencies of this BuildConfiguration?
        :param bool keep_pod_on_failure: Should we keep the build container running, if the build fails?
        :param bool timestamp_alignment: Should we add a timestamp during the alignment? Valid only for temporary builds.
        :param str rebuild_mode: Rebuild Modes: FORCE: always rebuild the configuration; EXPLICIT_DEPENDENCY_CHECK: check if any of user defined dependencies has been update; IMPLICIT_DEPENDENCY_CHECK: check if any captured dependency has been updated;
        :return: BuildRecordSingleton
                 If the method is called asynchronously,
                 returns the request thread.
        """
        kwargs['_return_http_data_only'] = True
        if kwargs.get('callback'):
            return self.trigger_with_http_info(id, **kwargs)
        else:
            (data) = self.trigger_with_http_info(id, **kwargs)
            return data