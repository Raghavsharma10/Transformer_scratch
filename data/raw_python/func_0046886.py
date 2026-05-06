def _parse_attributes(self, element_name, package_class, namespace=''):
        """
        Returns an instance of the package_class instantiated with a
        dictionary of the attributes from element_name in the specified
        namespace of the RSS feed.
        """
        return package_class(
            self._channel.find(
                './/{0}{1}'.format(namespace, element_name)
            ).attrib
        )