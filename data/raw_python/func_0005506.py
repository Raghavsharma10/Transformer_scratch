def set_scenario_network_info(self):
        """Populates a list of network info for each scenario host from
        deployment properties

        :return: None
        """
        log = logging.getLogger(self.cls_logger + '.set_scenario_network_info')

        for scenario_host in self.scenario_role_names:
            scenario_host_network_info = {'scenario_role_name': scenario_host}
            log.debug('Looking up network info from deployment properties for scenario host: {s}'.format(
                s=scenario_host))
            network_name_props = self.get_matching_property_names(
                'cons3rt.fap.deployment.machine.*{r}.*networkName'.format(r=scenario_host)
            )
            log.debug('Found {n} network name props'.format(n=str(len(network_name_props))))

            network_info_list = []
            for network_name_prop in network_name_props:
                network_info = {}
                network_name = self.get_value(network_name_prop)
                if not network_name:
                    log.debug('Network name not found for prop: {n}'.format(n=network_name_prop))
                    continue
                log.debug('Adding info for network name: {n}'.format(n=network_name))
                network_info['network_name'] = network_name
                interface_name_prop = 'cons3rt.fap.deployment.machine.{r}.{n}.interfaceName'.format(
                    r=scenario_host, n=network_name)
                interface_name = self.get_value(interface_name_prop)
                if interface_name:
                    network_info['interface_name'] = interface_name
                external_ip_prop = 'cons3rt.fap.deployment.machine.{r}.{n}.externalIp'.format(
                    r=scenario_host, n=network_name)
                external_ip = self.get_value(external_ip_prop)
                if external_ip:
                    network_info['external_ip'] = external_ip
                internal_ip_prop = 'cons3rt.fap.deployment.machine.{r}.{n}.internalIp'.format(
                    r=scenario_host, n=network_name)
                internal_ip = self.get_value(internal_ip_prop)
                if internal_ip:
                    network_info['internal_ip'] = internal_ip
                is_cons3rt_connection_prop = 'cons3rt.fap.deployment.machine.{r}.{n}.isCons3rtConnection'.format(
                    r=scenario_host, n=network_name)
                is_cons3rt_connection = self.get_value(is_cons3rt_connection_prop)
                if is_cons3rt_connection:
                    if is_cons3rt_connection.lower().strip() == 'true':
                        network_info['is_cons3rt_connection'] = True
                    else:
                        network_info['is_cons3rt_connection'] = False
                mac_address_prop = 'cons3rt.fap.deployment.machine.{r}.{n}.mac'.format(r=scenario_host, n=network_name)
                mac_address = self.get_value(mac_address_prop)
                if mac_address:
                    # Trim the escape characters from the mac address
                    mac_address = mac_address.replace('\\', '')
                    network_info['mac_address'] = mac_address
                log.debug('Found network info: {n}'.format(n=str(network_info)))
                network_info_list.append(network_info)
            scenario_host_network_info['network_info'] = network_info_list
            self.scenario_network_info.append(scenario_host_network_info)