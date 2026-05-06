def delete_old_host(self, hostname):
        """Remove all records for the host.

        :param str hostname: Hostname to remove
        :rtype: bool

        """
        host = Host(self.session, name=hostname)
        return host.delete()