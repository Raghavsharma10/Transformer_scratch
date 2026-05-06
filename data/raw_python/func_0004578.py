def assoc_ipv6(self, id_ip, id_equip, id_net):
        """
        Associate an IP6 with equipment.

        :param id_ip: IPv6 identifier.
        :param id_equip: Equipment identifier. Integer value and greater than zero.
        :param id_net: Network identifier. Integer value and greater than zero.

        :return: None

        :raise InvalidParameterError: IPv6, Equipment or Network identifier is none or invalid.
        :raise DataBaseError: Networkapi failed to access the database.
        :raise XMLError: Networkapi failed to generate the XML response.
        """

        if not is_valid_int_param(id_ip):
            raise InvalidParameterError(
                u'Ipv6 identifier is invalid or was not informed.')

        if not is_valid_int_param(id_net):
            raise InvalidParameterError(
                u'Network identifier is invalid or was not informed.')

        if not is_valid_int_param(id_equip):
            raise InvalidParameterError(
                u'Equipment identifier is invalid or was not informed.')

        ip_map = dict()
        ip_map['id_ip'] = id_ip
        ip_map['id_net'] = id_net
        ip_map['id_equip'] = id_equip

        url = "ipv6/assoc/"

        code, xml = self.submit({'ip_map': ip_map}, 'POST', url)

        return self.response(code, xml)