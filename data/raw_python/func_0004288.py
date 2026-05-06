def add(self, networkipv4_id, ipv4_id):
        """List all DHCPRelayIPv4.

        :param: networkipv4_id, ipv4_id

        :return: Following dictionary:
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
          }

        :raise NetworkAPIException: Falha ao acessar fonte de dados
        """

        data = dict()

        data['networkipv4'] = networkipv4_id
        data['ipv4'] = dict()
        data['ipv4']['id'] = ipv4_id
        uri = 'api/dhcprelayv4/'
        return self.post(uri, data=data)