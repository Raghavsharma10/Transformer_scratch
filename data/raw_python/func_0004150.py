def create(self, networkipv4s):
        """
        Method to create network-ipv4's

        :param networkipv4s: List containing networkipv4's desired to be created on database
        :return: None
        """

        data = {'networks': networkipv4s}
        return super(ApiNetworkIPv4, self).post('api/v3/networkv4/', data)