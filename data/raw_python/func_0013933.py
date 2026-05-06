def modify(self, sp=None, ip_port=None, ip_address=None, netmask=None,
               v6_prefix_length=None, gateway=None, vlan_id=None):
        """
        Modifies a replication interface.

        :param sp: same as the one in `create` method.
        :param ip_port: same as the one in `create` method.
        :param ip_address: same as the one in `create` method.
        :param netmask: same as the one in `create` method.
        :param v6_prefix_length: same as the one in `create` method.
        :param gateway: same as the one in `create` method.
        :param vlan_id: same as the one in `create` method.
        """
        req_body = self._cli.make_body(sp=sp, ipPort=ip_port,
                                       ipAddress=ip_address, netmask=netmask,
                                       v6PrefixLength=v6_prefix_length,
                                       gateway=gateway, vlanId=vlan_id)

        resp = self.action('modify', **req_body)
        resp.raise_if_err()
        return resp