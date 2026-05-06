def get_route_to(self, destination='', protocol=''):
        """Return route details to a specific destination, learned from a certain protocol."""

        # Note, it should be possible to query the FIB:
        # "<show><routing><fib></fib></routing></show>"
        # To add informations to this getter
        routes = {}

        if destination:
            destination = "<destination>{0}</destination>".format(destination)
        if protocol:
            protocol = "<type>{0}</type>".format(protocol)

        cmd = "<show><routing><route>{0}{1}</route></routing></show>".format(protocol, destination)
        try:
            self.device.op(cmd=cmd)
            routes_table_xml = xmltodict.parse(self.device.xml_root())
            routes_table_json = json.dumps(routes_table_xml['response']['result']['entry'])
            routes_table = json.loads(routes_table_json)
        except (AttributeError, KeyError):
            routes_table = []

        if isinstance(routes_table, dict):
            routes_table = [routes_table]

        for route in routes_table:
            d = {
                'current_active': False,
                'last_active': False,
                'age': -1,
                'next_hop': u'',
                'protocol': u'',
                'outgoing_interface': u'',
                'preference': -1,
                'inactive_reason': u'',
                'routing_table': u'default',
                'selected_next_hop': False,
                'protocol_attributes': {}
            }
            destination = route['destination']
            flags = route['flags']

            if 'A' in flags:
                d['current_active'] = True
            else:
                d['current_active'] = False
            if 'C' in flags:
                d['protocol'] = "connect"
            if 'S' in flags:
                d['protocol'] = "static"
            if 'R' in flags:
                d['protocol'] = "rip"
            if 'R' in flags:
                d['protocol'] = "rip"
            if 'O' in flags:
                d['protocol'] = "ospf"
            if 'B' in flags:
                d['protocol'] = "bgp"
            if 'H' in flags:
                d['protocol'] = "host"
            if route['age'] is not None:
                d['age'] = int(route['age'])
            if route['nexthop'] is not None:
                d['next_hop'] = route['nexthop']
            if route['interface'] is not None:
                d['outgoing_interface'] = route['interface']
            if route['metric'] is not None:
                d['preference'] = int(route['metric'])
            if route['virtual-router'] is not None:
                d['routing_table'] = route['virtual-router']

            if destination not in routes.keys():
                routes[destination] = []
            routes[destination].append(d)

        return routes