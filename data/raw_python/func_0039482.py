def addServerResource(self, pluginSubPath: bytes, resource: BasicResource) -> None:
        """ Add Server Resource

        Add a cusotom implementation of a served http resource.

        :param pluginSubPath: The resource path where you want to serve this resource.
        :param resource: The resource to serve.
        :return: None

        """
        pluginSubPath = pluginSubPath.strip(b'/')
        self.__rootServerResource.putChild(pluginSubPath, resource)