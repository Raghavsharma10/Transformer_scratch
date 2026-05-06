def zPushLens(self, update=None):
        """Push lens in ZOS COM server to UI"""
        self.SaveAs(self._sync_ui_file)
        OpticalSystem._dde_link.zLoadFile(self._sync_ui_file)
        OpticalSystem._dde_link.zPushLens(update)