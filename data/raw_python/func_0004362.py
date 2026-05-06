def associate_ipv6(self, id_equip, id_ipv6):
        """Associates an IPv6 to a equipament.

        :param id_equip: Identifier of the equipment. Integer value and greater than zero.
        :param id_ipv6: Identifier of the ip. Integer value and greater than zero.

        :return: Dictionary with the following structure:
            {'ip_equipamento': {'id': < id_ip_do_equipamento >}}

        :raise EquipamentoNaoExisteError: Equipment is not registered.
        :raise IpNaoExisteError: IP not registered.
        :raise IpError: IP is already associated with the equipment.
        :raise InvalidParameterError:  Identifier of the equipment and/or IP is null or invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_equip):
            raise InvalidParameterError(
                u'The identifier of equipment is invalid or was not informed.')

        if not is_valid_int_param(id_ipv6):
            raise InvalidParameterError(
                u'The identifier of ip is invalid or was not informed.')

        url = 'ipv6/' + str(id_ipv6) + '/equipment/' + str(id_equip) + '/'

        code, xml = self.submit(None, 'PUT', url)

        return self.response(code, xml)