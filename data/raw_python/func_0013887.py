def update_hosts(self, host_names):

        """Primarily for puppet-unity use.

        Update the hosts for the lun if needed.

        :param host_names: specify the new hosts which access the LUN.
        """

        if self.host_access:
            curr_hosts = [access.host.name for access in self.host_access]
        else:
            curr_hosts = []

        if set(curr_hosts) == set(host_names):
            log.info('Hosts for updating is equal to current hosts, '
                     'skip modification.')
            return None

        new_hosts = [UnityHostList.get(cli=self._cli, name=host_name)[0]
                     for host_name in host_names]
        new_access = [{'host': item,
                       'accessMask': HostLUNAccessEnum.PRODUCTION}
                      for item in new_hosts]
        resp = self.modify(host_access=new_access)
        resp.raise_if_err()
        return resp