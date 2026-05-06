def set_hosts_file_entry_for_role(self, role_name, network_name='user-net', fqdn=None, domain_name=None):
        """Adds an entry to the hosts file for a scenario host given
        the role name and network name

        :param role_name: (str) role name of the host to add
        :param network_name: (str) Name of the network to add to the hosts file
        :param fqdn: (str) Fully qualified domain name to use in the hosts file entry (trumps domain name)
        :param domain_name: (str) Domain name to include in the hosts file entries if provided
        :return:
        """
        log = logging.getLogger(self.cls_logger + '.set_hosts_file_entry_for_role')

        # Determine the host file entry portion
        if fqdn:
            host_file_entry = fqdn
        else:
            if domain_name:
                host_file_entry = '{r}.{d} {r}'.format(r=role_name, d=domain_name)
            else:
                host_file_entry = role_name
        log.info('Using hosts file entry: {e}'.format(e=host_file_entry))
        log.info('Scanning scenario hosts for role name [{r}] and network: {n}'.format(r=role_name, n=network_name))
        for scenario_host in self.scenario_network_info:
            if scenario_host['scenario_role_name'] == role_name:
                for host_network_info in scenario_host['network_info']:
                    if host_network_info['network_name'] == network_name:
                        self.update_hosts_file(ip=host_network_info['internal_ip'], entry=host_file_entry)