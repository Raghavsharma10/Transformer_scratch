def add(self, name, description):
        """Inserts a new Filter and returns its identifier.

        :param name: Name. String with a maximum of 100 characters and respect [a-zA-Z\_-]
        :param description: Description. String with a maximum of 200 characters and respect [a-zA-Z\_-]

        :return: Following dictionary:

        ::

            {'filter': {'id': < id >}}

        :raise InvalidParameterError: The value of name or description is invalid.
        :raise FilterDuplicateError: A filter named by name already exists.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        filter_map = dict()
        filter_map['name'] = name
        filter_map['description'] = description

        code, xml = self.submit({'filter': filter_map}, 'POST', 'filter/')

        return self.response(code, xml)