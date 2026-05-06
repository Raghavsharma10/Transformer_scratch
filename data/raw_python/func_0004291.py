def list(self, networkipv6=None, ipv6=None):
        """List all DHCPRelayIPv6.

        :param: networkipv6: networkipv6 id - list all dhcprelay filtering by networkipv6 id
          ipv6: ipv6 id - list all dhcprelay filtering by ipv6 id

        :return: Following dictionary:
          [
            {
            "networkipv6": <networkipv4_id>,
            "id": <id>,
            "ipv6": {
                "block1": <block1>,
                "block2": <block2>,
                "block3": <block3>,
                "block4": <block4>,
                "block5": <block5>,
                "block6": <block6>,
                "block7": <block7>,
                "block8": <block8>,
                "ip_formated": "<string IPv6>",
                "networkipv6": <networkipv6_id>,
                "id": <ipv6_id>,
                "description": "<string description>"
            },
            {...}
          ]

        :raise NetworkAPIException: Falha ao acessar fonte de dados
        """

        uri = 'api/dhcprelayv6/?'
        if networkipv6:
            uri += 'networkipv6=%s&' % networkipv6
        if ipv6:
            uri += 'ipv6=%s' % ipv6

        return self.get(uri)