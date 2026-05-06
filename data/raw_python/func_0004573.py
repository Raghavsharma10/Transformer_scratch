def save_ipv6(self, ip6, id_equip, descricao, id_net):
        """
        Save an IP6 and associate with equipment

        :param ip6: An IP6 available to save in format xxxx:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx:xxxx.
        :param id_equip: Equipment identifier. Integer value and greater than zero.
        :param descricao: IPv6 description.
        :param id_net: Network identifier. Integer value and greater than zero.

        :return: Dictionary with the following structure:

        ::

            {'ipv6': {'id': < id >,
            'block1': <block1>,
            'block2': <block2>,
            'block3': <block3>,
            'block4': <block4>,
            'block5': <block5>,
            'block6': <block6>,
            'block7': <block7>,
            'block8': <block8>,
            'descricao': < description >,
            'equipamento': [ { all name equipamentos related } ], }}
        """
        if not is_valid_int_param(id_net):
            raise InvalidParameterError(
                u'Network identifier is invalid or was not informed.')

        if not is_valid_int_param(id_equip):
            raise InvalidParameterError(
                u'Equipment identifier is invalid or was not informed.')

        if ip6 is None or ip6 == "":
            raise InvalidParameterError(
                u'IPv6 is invalid or was not informed.')

        ip_map = dict()
        ip_map['id_net'] = id_net
        ip_map['descricao'] = descricao
        ip_map['ip6'] = ip6
        ip_map['id_equip'] = id_equip

        url = "ipv6/save/"

        code, xml = self.submit({'ip_map': ip_map}, 'POST', url)

        return self.response(code, xml)