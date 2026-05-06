def set_scenario_hosts_file(self, network_name='user-net', domain_name=None):
        """Adds hosts file entries for each system in the scenario
        for the specified network_name provided

        :param network_name: (str) Name of the network to add to the hosts file
        :param domain_name: (str) Domain name to include in the hosts file entries if provided
        :return: None
        """
        log = logging.getLogger(self.cls_logger + '.set_scenario_hosts_file')

        log.info('Scanning scenario hosts to make entries in the hosts file for network: {n}'.format(n=network_name))
        for scenario_host in self.scenario_network_info:
            if domain_name:
                host_file_entry = '{r}.{d} {r}'.format(r=scenario_host['scenario_role_name'], d=domain_name)
            else:
                host_file_entry = scenario_host['scenario_role_name']
            for host_network_info in scenario_host['network_info']:
                if host_network_info['network_name'] == network_name:
                    self.update_hosts_file(ip=host_network_info['internal_ip'], entry=host_file_entry)