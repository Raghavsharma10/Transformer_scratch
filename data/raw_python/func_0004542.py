def create(self, networkipv6s):
        """
        Method to create network-ipv6's

        :param networkipv6s: List containing networkipv6's desired to be created on database
        :return: None
        """

        data = {'networks': networkipv6s}
        return super(ApiNetworkIPv6, self).post('api/v3/networkv6/', data)