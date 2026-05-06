def set_property(self, remote_path, option):
        """Sets metadata property of remote resource on WebDAV server.
        More information you can find by link http://webdav.org/specs/rfc4918.html#METHOD_PROPPATCH

        :param remote_path: the path to remote resource.
        :param option: the property attribute as dictionary with following keys:
                       `namespace`: (optional) the namespace for XML property which will be set,
                       `name`: the name of property which will be set,
                       `value`: (optional) the value of property which will be set. Defaults is empty string.
        """
        self.set_property_batch(remote_path=remote_path, option=[option])