def Save(self):
        """Saves the current system"""
        # This method is intercepted to allow ui_sync
        if self._file_to_save_on_Save:
            self._iopticalsystem.SaveAs(self._file_to_save_on_Save)
        else:
            self._iopticalsystem.Save()