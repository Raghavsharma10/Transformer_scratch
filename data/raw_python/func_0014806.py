def get_inventory_options(self):
        """Determine inventory options. Environment variables always
        take precedence over configuration files."""
        ip_type = self.config.get('inventory', 'inventory_ip_type')
        # If the appropriate environment variables are set, they override
        # other configuration
        ip_type = os.environ.get('INVENTORY_IP_TYPE', ip_type)
        return ip_type