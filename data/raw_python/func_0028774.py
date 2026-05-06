def SaveAs(self, filename):
        """Saves the current system to the specified file. 

        @param filename: absolute path (string)
        @return: None
        @raise: ValueError if path (excluding the zemax file name) is not valid

        All future calls to `Save()`  will use the same file.
        """
        directory, zfile = _os.path.split(filename)
        if zfile.startswith('pyzos_ui_sync_file'):
            self._iopticalsystem.SaveAs(filename)
        else: # regular file
            if not _os.path.exists(directory):
                raise ValueError('{} is not valid.'.format(directory))
            else:
                self._file_to_save_on_Save = filename   # store to use in Save()
                self._iopticalsystem.SaveAs(filename)