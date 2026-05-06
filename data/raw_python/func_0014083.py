def get_interfaces(self):
        """Return interfaces details."""
        result = {}

        interfaces = junos_views.junos_iface_table(self.device)
        interfaces.get()

        # convert all the tuples to our pre-defined dict structure
        for iface in interfaces.keys():
            result[iface] = {
                'is_up': interfaces[iface]['is_up'],
                'is_enabled': interfaces[iface]['is_enabled'],
                'description': (interfaces[iface]['description'] or u''),
                'last_flapped': float((interfaces[iface]['last_flapped'] or -1)),
                'mac_address': napalm_base.helpers.convert(
                    napalm_base.helpers.mac,
                    interfaces[iface]['mac_address'],
                    py23_compat.text_type(interfaces[iface]['mac_address'])),
                'speed': -1
            }
            # result[iface]['last_flapped'] = float(result[iface]['last_flapped'])

            match = re.search(r'(\d+)(\w*)', interfaces[iface]['speed'] or u'')
            if match is None:
                continue
            speed_value = napalm_base.helpers.convert(int, match.group(1), -1)
            if speed_value == -1:
                continue
            speed_unit = match.group(2)
            if speed_unit.lower() == 'gbps':
                speed_value *= 1000
            result[iface]['speed'] = speed_value

        return result