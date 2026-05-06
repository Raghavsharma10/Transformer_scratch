def search_ipv6_environment(self, ipv6, id_environment):
        """Get IPv6 with an associated environment.

        :param ipv6: IPv6 address in the format x1:x2:x3:x4:x5:x6:x7:x8.
        :param id_environment: Environment identifier. Integer value and greater than zero.

        :return: Dictionary with the following structure:

        ::


            {'ipv6': {'id': < id >,
            'id_vlan': < id_vlan >,
            'bloco1': < bloco1 >,
            'bloco2': < bloco2 >,
            'bloco3': < bloco3 >,
            'bloco4': < bloco4 >,
            'bloco5': < bloco5 >,
            'bloco6': < bloco6 >,
            'bloco7': < bloco7 >,
            'bloco8': < bloco8 >,
            'descricao': < descricao > }}

        :raise IpNaoExisteError: IPv6 is not registered or is not associated to the environment.
        :raise AmbienteNaoExisteError: Environment not found.
        :raise InvalidParameterError: Environment identifier and/or IPv6 string is/are none or invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_environment):
            raise InvalidParameterError(
                u'Environment identifier is invalid or was not informed.')

        ipv6_map = dict()
        ipv6_map['ipv6'] = ipv6
        ipv6_map['id_environment'] = id_environment

        code, xml = self.submit(
            {'ipv6_map': ipv6_map}, 'POST', 'ipv6/environment/')

        return self.response(code, xml)