def list(self, networkipv4=None, ipv4=None):
        """List all DHCPRelayIPv4.

        :param: networkipv4: networkipv4 id - list all dhcprelay filtering by networkipv4 id
          ipv4: ipv4 id - list all dhcprelay filtering by ipv4 id

        :return: Following dictionary:
          [
            {
            "networkipv4": <networkipv4_id>,
            "id": <id>,
            "ipv4": {
                "oct4": <oct4>,
                "oct2": <oct2>,
                "oct3": <oct3>,
                "oct1": <oct1>,
                "ip_formated": "<string IPv4>",
                "networkipv4": <networkipv4_id>,
                "id": <ipv4_id>,
                "descricao": "<string description>"
            },
            {...}
          ]

        :raise NetworkAPIException: Falha ao acessar fonte de dados
        """

        uri = 'api/dhcprelayv4/?'
        if networkipv4:
            uri += 'networkipv4=%s&' % networkipv4
        if ipv4:
            uri += 'ipv4=%s' % ipv4

        return self.get(uri)