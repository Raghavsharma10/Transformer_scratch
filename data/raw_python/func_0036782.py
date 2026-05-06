def set_related_method(self, resource, full_resource_url):
        """
        Using reflection, generate the related method and return it.
        """
        method_name = self.get_method_name(resource, 'get')

        def get(self, **kwargs):
            return self._call_api_single_related_resource(
                resource, full_resource_url, method_name, **kwargs
            )

        def get_list(self, **kwargs):
            return self._call_api_many_related_resources(
                resource, full_resource_url, method_name, **kwargs
            )

        if isinstance(full_resource_url, list):
            setattr(
                self, method_name,
                types.MethodType(get_list, self)
            )
        else:
            setattr(
                self, method_name,
                types.MethodType(get, self)
            )