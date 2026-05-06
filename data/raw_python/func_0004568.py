def save_ipv4(self, ip4, id_equip, descricao, id_net):
        """
        Save a IP4 and associate with equipment

        :param ip4: An IP4 available to save in format x.x.x.x.
        :param id_equip: Equipment identifier. Integer value and greater than zero.
        :param descricao: IP description.
        :param id_net: Network identifier. Integer value and greater than zero.

        :return: Dictionary with the following structure:

        ::

            { ip: { id: <id_ip4>,
            oct1: <oct1>,
            oct2: <oct2>,
            oct3: <oct3>,
            oct4: <oct4>,
            equipamento: [ { all equipamentos related } ] ,
            descricao: <descricao> } }
        """
        if not is_valid_int_param(id_net):
            raise InvalidParameterError(
                u'Network identifier is invalid or was not informed.')

        if not is_valid_int_param(id_equip):
            raise InvalidParameterError(
                u'Equipment identifier is invalid or was not informed.')

        if ip4 is None or ip4 == "":
            raise InvalidParameterError(u'IP4 is invalid or was not informed.')

        ip_map = dict()
        ip_map['id_net'] = id_net
        ip_map['descricao'] = descricao
        ip_map['ip4'] = ip4
        ip_map['id_equip'] = id_equip

        url = "ipv4/save/"

        code, xml = self.submit({'ip_map': ip_map}, 'POST', url)

        return self.response(code, xml)