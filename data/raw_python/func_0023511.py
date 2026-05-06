def get_property(self, remote_path, option):
        """Gets metadata property of remote resource on WebDAV server.
        More information you can find by link http://webdav.org/specs/rfc4918.html#METHOD_PROPFIND

        :param remote_path: the path to remote resource.
        :param option: the property attribute as dictionary with following keys:
                       `namespace`: (optional) the namespace for XML property which will be set,
                       `name`: the name of property which will be set.
        :return: the value of property or None if property is not found.
        """
        urn = Urn(remote_path)
        if not self.check(urn.path()):
            raise RemoteResourceNotFound(urn.path())

        data = WebDavXmlUtils.create_get_property_request_content(option)
        response = self.execute_request(action='get_property', path=urn.quote(), data=data)
        return WebDavXmlUtils.parse_get_property_response(response.content, option['name'])