def listar_healtchcheck_expect_distinct(self):
        """Get all expect_string.

        :return: Dictionary with the following structure:

        ::

            {'healthcheck_expect': [
             'expect_string': < expect_string >,
             ... demais healthcheck_expects ...]}


        :raise InvalidParameterError: Identifier is null and invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        url = 'healthcheckexpect/distinct/busca/'

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)