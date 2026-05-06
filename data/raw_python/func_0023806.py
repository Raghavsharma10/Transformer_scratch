def configure_firewall(self, server, firewall_rule_bodies):
        """
        Helper for calling create_firewall_rule in series for a list of firewall_rule_bodies.
        """
        server_uuid, server_instance = uuid_and_instance(server)

        return [
            self.create_firewall_rule(server_uuid, rule)
            for rule in firewall_rule_bodies
        ]