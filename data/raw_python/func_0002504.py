def _parse_connection_name(self, name):
        """
        Parse the connection into a tuple of the name and read / write type

        :param name: The name of the connection
        :type name: str

        :return: A tuple of the name and read / write type
        :rtype: tuple
        """
        if name is None:
            name = self.get_default_connection()

        if name.endswith(('::read', '::write')):
            return name.split('::', 1)

        return name, None