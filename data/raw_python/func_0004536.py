def undeploy(self, id_networkv6):
        """Remove deployment of network in equipments and set column 'active = 0' in tables redeipv6 ]

        :param id_networkv6: ID for NetworkIPv6

        :return: Equipments configuration output
        """

        uri = 'api/networkv6/%s/equipments/' % id_networkv6
        return super(ApiNetworkIPv6, self).delete(uri)