def update(self, networkipv4s):
        """
        Method to update network-ipv4's

        :param networkipv4s: List containing network-ipv4's desired to updated
        :return: None
        """

        data = {'networks': networkipv4s}
        networkipv4s_ids = [str(networkipv4.get('id'))
                            for networkipv4 in networkipv4s]

        return super(ApiNetworkIPv4, self).put('api/v3/networkv4/%s/' %
                                               ';'.join(networkipv4s_ids), data)