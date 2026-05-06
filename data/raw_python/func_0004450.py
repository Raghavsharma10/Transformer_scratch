def get(self, id_filter):
        """Get filter by id.

        :param id_filter: Identifier of the Filter. Integer value and greater than zero.

        :return: Following dictionary:

        ::

            {‘filter’: {‘id’: < id >,
            ‘name’: < name >,
            ‘description’: < description >}}

        :raise InvalidParameterError: The value of id_filter is invalid.
        :raise FilterNotFoundError: Filter not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """
        url = 'filter/get/' + str(id_filter) + '/'

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)