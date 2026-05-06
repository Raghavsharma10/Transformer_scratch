def get_resource_url(cls, resource, base_url):
        """
        Construct the URL for talking to this resource.

        i.e.:

        http://myapi.com/api/resource

        Note that this is NOT the method for calling individual instances i.e.

        http://myapi.com/api/resource/1

        Args:
            resource: The resource class instance
            base_url: The Base URL of this API service.
        returns:
            resource_url: The URL for this resource
        """
        if resource.Meta.resource_name:
            url = '{}/{}'.format(base_url, resource.Meta.resource_name)
        else:
            p = inflect.engine()
            plural_name = p.plural(resource.Meta.name.lower())
            url = '{}/{}'.format(base_url, plural_name)
        return cls._parse_url_and_validate(url)