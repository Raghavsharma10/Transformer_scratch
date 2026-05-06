def destroy(self):
        """
        Remove this FirewallRule from the API.

        This instance must be associated with a server for this method to work,
        which is done by instantiating via server.get_firewall_rules().
        """
        if not hasattr(self, 'server') or not self.server:
            raise Exception(
                """FirewallRule not associated with server;
                please use or server.get_firewall_rules() to get objects
                that are associated with a server.
                """)
        return self.server.cloud_manager.delete_firewall_rule(
            self.server.uuid,
            self.position
        )