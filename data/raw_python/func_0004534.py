def get_by_id(self, id_networkv6):
        """Get IPv6 network

        :param id_networkv4: ID for NetworkIPv6

        :return: IPv6 Network
        """

        uri = 'api/networkv4/%s/' % id_networkv6
        return super(ApiNetworkIPv6, self).get(uri)