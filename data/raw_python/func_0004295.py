def create(self, ipv6s):
        """
        Method to create ipv6's

        :param ipv6s: List containing vrf desired to be created on database
        :return: None
        """

        data = {'ips': ipv6s}
        return super(ApiIPv6, self).post('api/v3/ipv6/', data)