def add(self, networkipv6_id, ipv6_id):
        """List all DHCPRelayIPv4.

        :param: Object DHCPRelayIPv4

        :return: Following dictionary:
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
          }

        :raise NetworkAPIException: Falha ao acessar fonte de dados
        """

        data = dict()

        data['networkipv6'] = networkipv6_id
        data['ipv6'] = dict()
        data['ipv6']['id'] = ipv6_id
        uri = 'api/dhcprelayv6/'
        return self.post(uri, data=data)