def get_facts(self):
        """Return a set of facts from the devices."""
        # default values.
        vendor = u'Cisco'
        uptime = -1
        serial_number, fqdn, os_version, hostname, domain_name = ('',) * 5

        # obtain output from device
        show_ver = self.device.send_command('show version')
        show_hosts = self.device.send_command('show hosts')
        show_int_status = self.device.send_command('show interface status')
        show_hostname = self.device.send_command('show hostname')

        # uptime/serial_number/IOS version
        for line in show_ver.splitlines():
            if ' uptime is ' in line:
                _, uptime_str = line.split(' uptime is ')
                uptime = self.parse_uptime(uptime_str)

            if 'Processor Board ID' in line:
                _, serial_number = line.split("Processor Board ID ")
                serial_number = serial_number.strip()

            if 'system: ' in line:
                line = line.strip()
                os_version = line.split()[2]
                os_version = os_version.strip()

            if 'cisco' in line and 'Chassis' in line:
                _, model = line.split()[:2]
                model = model.strip()

        hostname = show_hostname.strip()

        # Determine domain_name and fqdn
        for line in show_hosts.splitlines():
            if 'Default domain' in line:
                _, domain_name = re.split(r".*Default domain.*is ", line)
                domain_name = domain_name.strip()
                break
        if hostname.count(".") >= 2:
            fqdn = hostname
        elif domain_name:
            fqdn = '{}.{}'.format(hostname, domain_name)

        # interface_list filter
        interface_list = []
        show_int_status = show_int_status.strip()
        for line in show_int_status.splitlines():
            if line.startswith(' ') or line.startswith('-') or line.startswith('Port '):
                continue
            interface = line.split()[0]
            interface_list.append(interface)

        return {
            'uptime': int(uptime),
            'vendor': vendor,
            'os_version': py23_compat.text_type(os_version),
            'serial_number': py23_compat.text_type(serial_number),
            'model': py23_compat.text_type(model),
            'hostname': py23_compat.text_type(hostname),
            'fqdn': fqdn,
            'interface_list': interface_list
        }