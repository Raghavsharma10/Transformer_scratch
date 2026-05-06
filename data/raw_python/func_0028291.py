def delvlan(self, vlanid):
        """
        Function operates on the IMCDev object. Takes input of vlanid (1-4094),
        auth and url to execute the delete_dev_vlans method on the IMCDev object. Device must be
        supported in the HPE IMC Platform VLAN Manager module.
        :param vlanid: str of VLANId ( valid 1-4094 )
        :return:
        """
        delete_dev_vlans( vlanid, self.auth, self.url, devid = self.devid)