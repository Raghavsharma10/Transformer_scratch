def update(self, networkipv6s):
        """
        Method to update network-ipv6's

        :param networkipv6s: List containing network-ipv6's desired to updated
        :return: None
        """

        data = {'networks': networkipv6s}
        networkipv6s_ids = [str(networkipv6.get('id'))
                            for networkipv6 in networkipv6s]

        return super(ApiNetworkIPv6, self).put('api/v3/networkv6/%s/' %
                                               ';'.join(networkipv6s_ids), data)