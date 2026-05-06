def get_ntp_servers(self):
        """Return the NTP servers configured on the device."""
        ntp_table = junos_views.junos_ntp_servers_config_table(self.device)
        ntp_table.get()

        ntp_servers = ntp_table.items()

        if not ntp_servers:
            return {}

        return {napalm_base.helpers.ip(server[0]): {} for server in ntp_servers}