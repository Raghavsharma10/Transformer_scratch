def create_firewall_rule(self, server, firewall_rule_body):
        """
        Create a new firewall rule for a given server uuid.

        The rule can begiven as a dict or with FirewallRule.prepare_post_body().
        Returns a FirewallRule object.
        """
        server_uuid, server_instance = uuid_and_instance(server)

        url = '/server/{0}/firewall_rule'.format(server_uuid)
        body = {'firewall_rule': firewall_rule_body}
        res = self.post_request(url, body)

        return FirewallRule(server=server_instance, **res['firewall_rule'])