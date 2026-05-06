def get_ntp_peers(self):
        """Return the NTP peers configured on the device."""
        ntp_table = junos_views.junos_ntp_peers_config_table(self.device)
        ntp_table.get()

        ntp_peers = ntp_table.items()

        if not ntp_peers:
            return {}

        return {napalm_base.helpers.ip(peer[0]): {} for peer in ntp_peers}