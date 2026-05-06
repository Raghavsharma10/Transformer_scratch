def deploy(self, id_networkv6):
        """Deploy network in equipments and set column 'active = 1' in tables redeipv6 ]

        :param id_networkv6: ID for NetworkIPv6

        :return: Equipments configuration output
        """

        data = dict()
        uri = 'api/networkv6/%s/equipments/' % id_networkv6

        return super(ApiNetworkIPv6, self).post(uri, data=data)