def remove(self, id_filter):
        """Remove Filter by the identifier.

        :param id_filter: Identifier of the Filter. Integer value and greater than zero.

        :return: None

        :raise InvalidParameterError: Filter identifier is null and invalid.
        :raise FilterNotFoundError: Filter not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_filter):
            raise InvalidParameterError(
                u'The identifier of Filter is invalid or was not informed.')

        url = 'filter/' + str(id_filter) + '/'

        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)