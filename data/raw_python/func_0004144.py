def undeploy(self, id_networkv4):
        """Remove deployment of network in equipments and set column 'active = 0' in tables redeipv4 ]

        :param id_networkv4: ID for NetworkIPv4

        :return: Equipments configuration output
        """

        uri = 'api/networkv4/%s/equipments/' % id_networkv4

        return super(ApiNetworkIPv4, self).delete(uri)