def configure_firewall(self, FirewallRules):
        """
        Helper function for automatically adding several FirewallRules in series.
        """
        firewall_rule_bodies = [
            FirewallRule.to_dict()
            for FirewallRule in FirewallRules
        ]
        return self.cloud_manager.configure_firewall(self, firewall_rule_bodies)