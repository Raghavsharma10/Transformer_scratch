def remove_vlan(self, vlan_resource_id):
        """
        Remove a VLAN
        :param vlan_resource_id:
        :return:
        """
        vlan_id = {'VLanResourceId': vlan_resource_id}
        json_scheme = self.gen_def_json_scheme('SetRemoveVLan', vlan_id)
        json_obj = self.call_method_post(method='SetRemoveVLan', json_scheme=json_scheme)
        return True if json_obj['Success'] is True else False