def get_firewall_rules(self, server):
        """
        Return all FirewallRule objects based on a server instance or uuid.
        """
        server_uuid, server_instance = uuid_and_instance(server)

        url = '/server/{0}/firewall_rule'.format(server_uuid)
        res = self.get_request(url)

        return [
            FirewallRule(server=server_instance, **firewall_rule)
            for firewall_rule in res['firewall_rules']['firewall_rule']
        ]