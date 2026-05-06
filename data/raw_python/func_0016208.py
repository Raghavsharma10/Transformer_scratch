def remove_ip(self, ip_id):
        """
        Delete an Ip from the boughs ip list
        @param (str) ip_id: a string representing the resource id of the IP
        @return: True if json method had success else False
        """
        ip_id = '    "IpAddressResourceId": %s' % ip_id
        json_scheme = self.gen_def_json_scheme('SetRemoveIpAddress', ip_id)
        json_obj = self.call_method_post(method='SetRemoveIpAddress', json_scheme=json_scheme)
        pprint(json_obj)
        return True if json_obj['Success'] is True else False