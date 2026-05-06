def create_networks(self, ids, id_vlan):
        """Set column 'active = 1' in tables redeipv4 and redeipv6]

        :param ids: ID for NetworkIPv4 and/or NetworkIPv6

        :return: Nothing
        """

        network_map = dict()
        network_map['ids'] = ids
        network_map['id_vlan'] = id_vlan

        code, xml = self.submit(
            {'network': network_map}, 'PUT', 'network/create/')

        return self.response(code, xml)