def get_by_id(self, id_networkv4):
        """Get IPv4 network

        :param id_networkv4: ID for NetworkIPv4

        :return: IPv4 Network
        """

        uri = 'api/networkv4/%s/' % id_networkv4

        return super(ApiNetworkIPv4, self).get(uri)