def edit_ipv6(self, ip6, descricao, id_ip):
        """
        Edit a IP6

        :param ip6: An IP6 available to save in format xxxx:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx.
        :param descricao: IP description.
        :param id_ip: Ipv6 identifier. Integer value and greater than zero.

        :return: None
        """

        if not is_valid_int_param(id_ip):
            raise InvalidParameterError(
                u'Ipv6 identifier is invalid or was not informed.')

        if ip6 is None or ip6 == "":
            raise InvalidParameterError(u'IP6 is invalid or was not informed.')

        ip_map = dict()
        ip_map['descricao'] = descricao
        ip_map['ip6'] = ip6
        ip_map['id_ip'] = id_ip

        url = "ipv6/edit/"

        code, xml = self.submit({'ip_map': ip_map}, 'POST', url)

        return self.response(code, xml)