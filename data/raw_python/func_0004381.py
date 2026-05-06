def buscar_healthcheck_por_id(self, id_healthcheck):
        """Get HealthCheck by id.

        :param id_healthcheck: HealthCheck ID.

        :return: Following dictionary:

        ::

            {'healthcheck_expect': {'match_list': < match_list >,
             'expect_string': < expect_string >,
             'id': < id >,
             'ambiente': < ambiente >}}

        :raise HealthCheckNaoExisteError:  HealthCheck not registered.
        :raise InvalidParameterError: HealthCheck identifier is null and invalid.
        :raise DataBaseError: Can't connect to networkapi database.
        :raise XMLError: Failed to generate the XML response.
        """
        if not is_valid_int_param(id_healthcheck):
            raise InvalidParameterError(
                u'O identificador do healthcheck é inválido ou não foi informado.')

        url = 'healthcheckexpect/get/' + str(id_healthcheck) + '/'

        code, xml = self.submit(None, 'GET', url)

        return self.response(code, xml)