def get_lldp_neighbors(self):
        """Return LLDP neighbors details."""

        neighbors = {}

        cmd = '<show><lldp><neighbors>all</neighbors></lldp></show>'
        try:
            self.device.op(cmd=cmd)
            lldp_table_xml = xmltodict.parse(self.device.xml_root())
            lldp_table_json = json.dumps(lldp_table_xml['response']['result']['entry'])
            lldp_table = json.loads(lldp_table_json)
        except AttributeError:
            lldp_table = []

        for lldp_item in lldp_table:

            local_int = lldp_item['@name']

            if local_int not in neighbors.keys():
                neighbors[local_int] = []
            try:
                lldp_neighs = lldp_item.get('neighbors').get('entry')
            except AttributeError:
                lldp_neighs = ''
            if isinstance(lldp_neighs, dict):
                lldp_neighs = [lldp_neighs]

            for neighbor in lldp_neighs:
                n = {}
                n['hostname'] = neighbor['system-name']
                n['port'] = neighbor['port-id']
                neighbors[local_int].append(n)
        return neighbors