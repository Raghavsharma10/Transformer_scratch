def delete_ipv4(self, ipv4_id):
        """
        Delete ipv4
        """
        uri = 'api/ipv4/%s/' % (ipv4_id)

        return super(ApiNetworkIPv4, self).delete(uri)