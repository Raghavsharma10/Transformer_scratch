def get_facts(self):
        """Return a set of facts from the devices."""
        # default values.
        vendor = u'Cisco'
        uptime = -1
        serial_number, fqdn, os_version, hostname, domain_name = ('Unknown',) * 5

        # obtain output from device
        show_ver = self._send_command('show version')
        show_hosts = self._send_command('show hosts')
        show_ip_int_br = self._send_command('show ip interface brief')

        # uptime/serial_number/IOS version
        for line in show_ver.splitlines():
            if ' uptime is ' in line:
                hostname, uptime_str = line.split(' uptime is ')
                uptime = self.parse_uptime(uptime_str)
                hostname = hostname.strip()

            if 'Processor board ID' in line:
                _, serial_number = line.split("Processor board ID ")
                serial_number = serial_number.strip()

            if re.search(r"Cisco IOS Software", line):
                try:
                    _, os_version = line.split("Cisco IOS Software, ")
                except ValueError:
                    # Handle 'Cisco IOS Software [Denali],'
                    _, os_version = re.split(r"Cisco IOS Software \[.*?\], ", line)
                os_version = os_version.strip()
            elif re.search(r"IOS (tm).+Software", line):
                _, os_version = line.split("IOS (tm) ")
                os_version = os_version.strip()

        # Determine domain_name and fqdn
        for line in show_hosts.splitlines():
            if 'Default domain' in line:
                _, domain_name = line.split("Default domain is ")
                domain_name = domain_name.strip()
                break
        if domain_name != 'Unknown' and hostname != 'Unknown':
            fqdn = u'{}.{}'.format(hostname, domain_name)

        # model filter
        try:
            match_model = re.search(r"Cisco (.+?) .+bytes of", show_ver, flags=re.IGNORECASE)
            model = match_model.group(1)
        except AttributeError:
            model = u'Unknown'

        # interface_list filter
        interface_list = []
        show_ip_int_br = show_ip_int_br.strip()
        for line in show_ip_int_br.splitlines():
            if 'Interface ' in line:
                continue
            interface = line.split()[0]
            interface_list.append(interface)

        return {
            'uptime': uptime,
            'vendor': vendor,
            'os_version': py23_compat.text_type(os_version),
            'serial_number': py23_compat.text_type(serial_number),
            'model': py23_compat.text_type(model),
            'hostname': py23_compat.text_type(hostname),
            'fqdn': fqdn,
            'interface_list': interface_list
        }