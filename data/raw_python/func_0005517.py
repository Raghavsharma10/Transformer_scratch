def get_device_for_network_linux(self, network_name):
        """Given a cons3rt network name, return the network interface name
        on this Linux system

        :param network_name: (str) Name of the network to search for
        :return: (str) name of the network interface device or None
        """
        log = logging.getLogger(self.cls_logger + '.get_device_for_network_linux')

        if get_os() not in ['Linux']:
            log.warn('Non-linux OS detected, returning...')
            return

        # Get the IP address for the network name according to cons3rt
        ip_address = self.get_ip_on_network(network_name=network_name)
        if not ip_address:
            log.warn('IP address not found for network with name: {n}'.format(n=network_name))
            return

        # Get the system device names and ip addresses
        sys_info = ip_addr()

        # Check for a matching IP address
        device_name = None
        for device_name, sys_ip_address in sys_info.iteritems():
            if sys_ip_address == ip_address:
                log.debug('Found matching system IP [{i}] for device: {d}'.format(i=ip_address, d=device_name))

        if not device_name:
            log.warn('Network device not found with IP address {i} in system network data: {d}'.format(
                i=ip_address, d=str(sys_info)))
            return
        log.debug('Found device name [{d}] with IP address [{i}] for network: {n}'.format(
            d=device_name, i=ip_address, n=network_name))
        return device_name