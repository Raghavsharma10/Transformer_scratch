def delete_ipv6(self, ipv6_id):
        """
        Delete ipv6
        """
        uri = 'api/ipv6/%s/' % (ipv6_id)

        return super(ApiNetworkIPv6, self).delete(uri)