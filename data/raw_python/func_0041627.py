def addDesktopResource(self, pluginSubPath: bytes, resource: BasicResource) -> None:
        """ Add Site Resource

        Add a cusotom implementation of a served http resource.

        :param pluginSubPath: The resource path where you want to serve this resource.
        :param resource: The resource to serve.
        :return: None

        """
        pluginSubPath = pluginSubPath.strip(b'/')
        self.__rootDesktopResource.putChild(pluginSubPath, resource)