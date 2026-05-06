def create_replication_interface(self, sp, ip_port, ip_address,
                                     netmask=None, v6_prefix_length=None,
                                     gateway=None, vlan_id=None):
        """
        Creates a replication interface.

        :param sp: `UnityStorageProcessor` object. Storage processor on which
            the replication interface is running.
        :param ip_port: `UnityIpPort` object. Physical port or link aggregation
            on the storage processor on which the interface is running.
        :param ip_address: IP address of the replication interface.
        :param netmask: IPv4 netmask for the replication interface, if it uses
            an IPv4 address.
        :param v6_prefix_length: IPv6 prefix length for the interface, if it
            uses an IPv6 address.
        :param gateway: IPv4 or IPv6 gateway address for the replication
            interface.
        :param vlan_id: VLAN identifier for the interface.
        :return: the newly create replication interface.
        """
        return UnityReplicationInterface.create(
            self._cli, sp, ip_port, ip_address, netmask=netmask,
            v6_prefix_length=v6_prefix_length, gateway=gateway,
            vlan_id=vlan_id)