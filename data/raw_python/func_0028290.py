def addvlan(self, vlanid, vlan_name):
        """
        Function operates on the IMCDev object. Takes input of vlanid (1-4094), str of vlan_name,
        auth and url to execute the create_dev_vlan method on the IMCDev object. Device must be
        supported in the HPE IMC Platform VLAN Manager module.
        :param vlanid: str of VLANId ( valid 1-4094 )
        :param vlan_name: str of vlan_name
        :return:
        """
        create_dev_vlan( vlanid, vlan_name, self.auth, self.url, devid = self.devid)