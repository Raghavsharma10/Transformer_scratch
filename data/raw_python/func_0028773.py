def zGetRefresh(self):
        """Copy lens in UI to headless ZOS COM server"""
        OpticalSystem._dde_link.zGetRefresh()
        OpticalSystem._dde_link.zSaveFile(self._sync_ui_file)
        self._iopticalsystem.LoadFile (self._sync_ui_file, False)