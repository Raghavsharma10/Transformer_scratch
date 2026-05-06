def alter(self, id_filter, name, description):
        """Change Filter by the identifier.

        :param id_filter: Identifier of the Filter. Integer value and greater than zero.
        :param name: Name. String with a maximum of 50 characters and respect [a-zA-Z\_-]
        :param description: Description. String with a maximum of 50 characters and respect [a-zA-Z\_-]

        :return: None

        :raise InvalidParameterError: Filter identifier is null and invalid.
        :raise InvalidParameterError: The value of name or description is invalid.
        :raise FilterNotFoundError: Filter not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_filter):
            raise InvalidParameterError(
                u'The identifier of Filter is invalid or was not informed.')

        filter_map = dict()
        filter_map['name'] = name
        filter_map['description'] = description

        url = 'filter/' + str(id_filter) + '/'

        code, xml = self.submit({'filter': filter_map}, 'PUT', url)

        return self.response(code, xml)