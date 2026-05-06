def remover_ip(self, id_equipamento, id_ip):
        """Removes association of IP and Equipment.
        If IP has no other association with equipments, IP is also removed.

        :param id_equipamento: Equipment identifier
        :param id_ip: IP identifier.

        :return: None

        :raise VipIpError: Ip can't be removed because there is a created Vip Request.
        :raise IpEquipCantDissociateFromVip: Equipment is the last balancer for created Vip Request.
        :raise IpError: IP not associated with equipment.
        :raise InvalidParameterError: Equipment or IP identifier is none or invalid.
        :raise EquipamentoNaoExisteError: Equipment doesn't exist.
        :raise DataBaseError: Networkapi failed to access database.
        :raise XMLError: Networkapi failed to build response XML.
        """

        if not is_valid_int_param(id_equipamento):
            raise InvalidParameterError(
                u'O identificador do equipamento é inválido ou não foi informado.')

        if not is_valid_int_param(id_ip):
            raise InvalidParameterError(
                u'O identificador do ip é inválido ou não foi informado.')

        url = 'ip/' + str(id_ip) + '/equipamento/' + str(id_equipamento) + '/'

        code, xml = self.submit(None, 'DELETE', url)

        return self.response(code, xml)