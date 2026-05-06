def set_property_batch(self, remote_path, option):
        """Sets batch metadata properties of remote resource on WebDAV server in batch.
        More information you can find by link http://webdav.org/specs/rfc4918.html#METHOD_PROPPATCH

        :param remote_path: the path to remote resource.
        :param option: the property attributes as list of dictionaries with following keys:
                       `namespace`: (optional) the namespace for XML property which will be set,
                       `name`: the name of property which will be set,
                       `value`: (optional) the value of property which will be set. Defaults is empty string.
        """
        urn = Urn(remote_path)
        if not self.check(urn.path()):
            raise RemoteResourceNotFound(urn.path())

        data = WebDavXmlUtils.create_set_property_batch_request_content(option)
        self.execute_request(action='set_property', path=urn.quote(), data=data)