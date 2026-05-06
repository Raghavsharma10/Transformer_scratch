def add(self, *resources):
        """
        Apply the tag to one or more resources

        :param resources: one or more `Resource` objects to which tags can be
            applied
        :return: `None`
        :raises DOAPIError: if the API endpoint replies with an error
        """
        self.doapi_manager.request(self.url + '/resources', method='POST',
                                   data={"resources": _to_taggable(resources)})