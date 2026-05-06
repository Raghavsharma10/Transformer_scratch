def get_interfaces_ip(self):
        '''Return IP interface data.'''

        def extract_ip_info(parsed_intf_dict):
            '''
            IPv4:
              - Primary IP is in the '<ip>' tag. If no v4 is configured the return value is 'N/A'.
              - Secondary IP's are in '<addr>'. If no secondaries, this field is not returned by
                the xmltodict.parse() method.

            IPv6:
              - All addresses are returned in '<addr6>'. If no v6 configured, this is not returned
                either by xmltodict.parse().

            Example of XML response for an intf with multiple IPv4 and IPv6 addresses:

            <response status="success">
              <result>
                <ifnet>
                  <entry>
                    <name>ethernet1/5</name>
                    <zone/>
                    <fwd>N/A</fwd>
                    <vsys>1</vsys>
                    <dyn-addr/>
                    <addr6>
                      <member>fe80::d61d:71ff:fed8:fe14/64</member>
                      <member>2001::1234/120</member>
                    </addr6>
                    <tag>0</tag>
                    <ip>169.254.0.1/30</ip>
                    <id>20</id>
                    <addr>
                      <member>1.1.1.1/28</member>
                    </addr>
                  </entry>
                  {...}
                </ifnet>
                <hw>
                  {...}
                </hw>
              </result>
            </response>
            '''
            intf = parsed_intf_dict['name']
            _ip_info = {intf: {}}

            v4_ip = parsed_intf_dict.get('ip')
            secondary_v4_ip = parsed_intf_dict.get('addr')
            v6_ip = parsed_intf_dict.get('addr6')

            if v4_ip != 'N/A':
                address, pref = v4_ip.split('/')
                _ip_info[intf].setdefault('ipv4', {})[address] = {'prefix_length': int(pref)}

            if secondary_v4_ip is not None:
                members = secondary_v4_ip['member']
                if not isinstance(members, list):
                    # If only 1 secondary IP is present, xmltodict converts field to a string, else
                    # it converts it to a list of strings.
                    members = [members]
                for address in members:
                    address, pref = address.split('/')
                    _ip_info[intf].setdefault('ipv4', {})[address] = {'prefix_length': int(pref)}

            if v6_ip is not None:
                members = v6_ip['member']
                if not isinstance(members, list):
                    # Same "1 vs many -> string vs list of strings" comment.
                    members = [members]
                for address in members:
                    address, pref = address.split('/')
                    _ip_info[intf].setdefault('ipv6', {})[address] = {'prefix_length': int(pref)}

            # Reset dictionary if no addresses were found.
            if _ip_info == {intf: {}}:
                _ip_info = {}

            return _ip_info

        ip_interfaces = {}
        cmd = "<show><interface>all</interface></show>"

        self.device.op(cmd=cmd)
        interface_info_xml = xmltodict.parse(self.device.xml_root())
        interface_info_json = json.dumps(
            interface_info_xml['response']['result']['ifnet']['entry']
        )
        interface_info = json.loads(interface_info_json)

        if isinstance(interface_info, dict):
            # Same "1 vs many -> dict vs list of dicts" comment.
            interface_info = [interface_info]

        for interface_dict in interface_info:
            ip_info = extract_ip_info(interface_dict)
            if ip_info:
                ip_interfaces.update(ip_info)

        return ip_interfaces