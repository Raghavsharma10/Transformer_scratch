def getipmacarp(self):
        """
        Function operates on the IMCDev object and updates the ipmacarp attribute
        :return:
        """
        self.ipmacarp = get_ip_mac_arp_list(self.auth, self.url, devid = self.devid)