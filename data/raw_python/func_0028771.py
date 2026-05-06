def zSyncWithUI(self):
        """Turn on sync-with-ui"""
        if not OpticalSystem._dde_link:
            OpticalSystem._dde_link = _get_new_dde_link()
        if not self._sync_ui_file:
            self._sync_ui_file = _get_sync_ui_filename()
        self._sync_ui = True