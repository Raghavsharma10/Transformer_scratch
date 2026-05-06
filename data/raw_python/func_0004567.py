def edit_ipv4(self, ip4, descricao, id_ip):
        """
        Edit a IP4

        :param ip4: An IP4 available to save in format x.x.x.x.
        :param id_ip: IP identifier. Integer value and greater than zero.
        :param descricao: IP description.

        :return: None
        """

        if not is_valid_int_param(id_ip):
            raise InvalidParameterError(
                u'Ip identifier is invalid or was not informed.')

        if ip4 is None or ip4 == "":
            raise InvalidParameterError(
                u'The IP4 is invalid or was not informed.')

        ip_map = dict()
        ip_map['descricao'] = descricao
        ip_map['ip4'] = ip4
        ip_map['id_ip'] = id_ip

        url = "ip4/edit/"

        code, xml = self.submit({'ip_map': ip_map}, 'POST', url)

        return self.response(code, xml)