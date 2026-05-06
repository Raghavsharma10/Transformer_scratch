def create_vlan(self, id_vlan):
        """ Set column 'ativada = 1'.

        :param id_vlan: VLAN identifier.

        :return: None
        """

        vlan_map = dict()

        vlan_map['vlan_id'] = id_vlan

        code, xml = self.submit({'vlan': vlan_map}, 'PUT', 'vlan/create/')

        return self.response(code, xml)