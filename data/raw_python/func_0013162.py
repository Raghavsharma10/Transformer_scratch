def _load_candidate_wrapper(self, source_file=None, source_config=None, dest_file=None,
                                file_system=None):
        """
        Transfer file to remote device for either merge or replace operations

        Returns (return_status, msg)
        """
        return_status = False
        msg = ''
        if source_file and source_config:
            raise ValueError("Cannot simultaneously set source_file and source_config")

        if source_config:
            if self.inline_transfer:
                (return_status, msg) = self._inline_tcl_xfer(source_config=source_config,
                                                             dest_file=dest_file,
                                                             file_system=file_system)
            else:
                # Use SCP
                tmp_file = self._create_tmp_file(source_config)
                (return_status, msg) = self._scp_file(source_file=tmp_file, dest_file=dest_file,
                                                      file_system=file_system)
                if tmp_file and os.path.isfile(tmp_file):
                    os.remove(tmp_file)
        if source_file:
            if self.inline_transfer:
                (return_status, msg) = self._inline_tcl_xfer(source_file=source_file,
                                                             dest_file=dest_file,
                                                             file_system=file_system)
            else:
                (return_status, msg) = self._scp_file(source_file=source_file, dest_file=dest_file,
                                                      file_system=file_system)
        if not return_status:
            if msg == '':
                msg = "Transfer to remote device failed"
        return (return_status, msg)