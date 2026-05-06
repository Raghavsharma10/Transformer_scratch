def _cleanup_ca_temp_file(self):
        """
        Function to clean up ca temp file for requests.

        **Returns:** Removes TEMP ca file, no return
        """
        if os.name == 'nt':
            if isinstance(self.ca_verify_filename, (binary_type, text_type)):
                # windows requires file to be closed for access. Have to manually remove
                os.unlink(self.ca_verify_filename)
        else:
            # other OS's allow close and delete of file.
            self._ca_verify_file_handle.close()