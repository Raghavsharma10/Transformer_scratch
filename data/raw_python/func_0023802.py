def get_firewall_rule(self, server_uuid, firewall_rule_position, server_instance=None):
        """
        Return a FirewallRule object based on server uuid and rule position.
        """
        url = '/server/{0}/firewall_rule/{1}'.format(server_uuid, firewall_rule_position)
        res = self.get_request(url)
        return FirewallRule(**res['firewall_rule'])