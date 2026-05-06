def deallocate_ip(self, hostipaddress):
        """
        Object method takes in input of hostip address,removes them from the parent ip scope.
        :param hostid: str of the hostid of  the target host ip record

        :return:
        """
        delete_host_from_segment(hostipaddress, self.netaddr, self.auth, self.url)