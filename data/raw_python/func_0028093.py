def _get_resource_from_obj(self, resource):
        # type: (Union[hdx.data.resource.Resource,Dict,str]) -> hdx.data.resource.Resource
        """Add new or update existing resource in dataset with new metadata

        Args:
            resource (Union[hdx.data.resource.Resource,Dict,str]): Either resource id or resource metadata from a Resource object or a dictionary

        Returns:
            hdx.data.resource.Resource: Resource object
        """
        if isinstance(resource, str):
            if is_valid_uuid(resource) is False:
                raise HDXError('%s is not a valid resource id!' % resource)
            resource = hdx.data.resource.Resource.read_from_hdx(resource, configuration=self.configuration)
        elif isinstance(resource, dict):
            resource = hdx.data.resource.Resource(resource, configuration=self.configuration)
        if not isinstance(resource, hdx.data.resource.Resource):
            raise HDXError('Type %s cannot be added as a resource!' % type(resource).__name__)
        return resource