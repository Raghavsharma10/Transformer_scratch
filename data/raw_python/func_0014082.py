def get_facts(self):
        """Return facts of the device."""
        output = self.device.facts

        uptime = self.device.uptime or -1

        interfaces = junos_views.junos_iface_table(self.device)
        interfaces.get()
        interface_list = interfaces.keys()

        return {
            'vendor': u'Juniper',
            'model': py23_compat.text_type(output['model']),
            'serial_number': py23_compat.text_type(output['serialnumber']),
            'os_version': py23_compat.text_type(output['version']),
            'hostname': py23_compat.text_type(output['hostname']),
            'fqdn': py23_compat.text_type(output['fqdn']),
            'uptime': uptime,
            'interface_list': interface_list
        }