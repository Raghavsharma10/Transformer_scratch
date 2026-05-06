def create(self, ipv6s):
        """
        Method to create ipv6's

        :param ipv6s: List containing vrf desired to be created on database
        :return: None
        """

        data = {'ips': ipv6s}
        return super(ApiV4IPv6, self).post('api/v4/ipv6/', data)