def deploy(self, id_networkv4):
        """Deploy network in equipments and set column 'active = 1' in tables redeipv4

        :param id_networkv4: ID for NetworkIPv4

        :return: Equipments configuration output
        """

        data = dict()
        uri = 'api/networkv4/%s/equipments/' % id_networkv4

        return super(ApiNetworkIPv4, self).post(uri, data=data)