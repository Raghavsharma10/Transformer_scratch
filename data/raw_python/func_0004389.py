def add_expect_string_healthcheck(self, expect_string):
        """Inserts a new healthckeck_expect  with only expect_string.

        :param expect_string: expect_string.

        :return: Dictionary with the following structure:

        ::

            {'healthcheck_expect': {'id': < id >}}

        :raise InvalidParameterError: The value of expect_string is invalid.
        :raise HealthCheckExpectJaCadastradoError: There is already a healthcheck_expect registered with the same data.
        :raise HealthCheckExpectNaoExisteError: Healthcheck_expect not registered.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.

        """

        healthcheck_map = dict()
        healthcheck_map['expect_string'] = expect_string

        url = 'healthcheckexpect/add/expect_string/'

        code, xml = self.submit({'healthcheck': healthcheck_map}, 'POST', url)

        return self.response(code, xml)