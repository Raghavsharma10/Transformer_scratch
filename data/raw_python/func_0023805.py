def delete_firewall_rule(self, server_uuid, firewall_rule_position):
        """
        Delete a firewall rule based on a server uuid and rule position.
        """
        url = '/server/{0}/firewall_rule/{1}'.format(server_uuid, firewall_rule_position)
        return self.request('DELETE', url)