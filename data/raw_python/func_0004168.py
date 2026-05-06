def apply_acl(self, equipments, vlan, environment, network):
        '''Apply the file acl in equipments

        :param equipments: list of equipments
        :param vlan: Vvlan
        :param environment: Environment
        :param network: v4 or v6

        :raise Exception: Failed to apply acl

        :return: True case Apply and sysout of script
        '''

        vlan_map = dict()
        vlan_map['equipments'] = equipments
        vlan_map['vlan'] = vlan
        vlan_map['environment'] = environment
        vlan_map['network'] = network

        url = 'vlan/apply/acl/'

        code, xml = self.submit({'vlan': vlan_map}, 'POST', url)

        return self.response(code, xml)