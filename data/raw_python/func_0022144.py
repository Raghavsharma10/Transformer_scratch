def ssl_verify(self, ssl_verify):
        """
        Modify ssl verification settings

        **Parameters:**

          - ssl_verify:
             - True: Verify using builtin BYTE_CA_BUNDLE.
             - False: No SSL Verification.
             - Str: Full path to a x509 PEM CA File or bundle.

        **Returns:** Mutates API object in place, no return.
        """
        self.verify = ssl_verify
        # if verify true/false, set ca_verify_file appropriately
        if isinstance(self.verify, bool):
            if self.verify:  # True
                if os.name == 'nt':
                    # Windows does not allow tmpfile access w/out close. Close file then delete it when done.
                    self._ca_verify_file_handle = temp_ca_bundle(delete=False)
                    self._ca_verify_file_handle.write(BYTE_CA_BUNDLE)
                    self._ca_verify_file_handle.flush()
                    self.ca_verify_filename = self._ca_verify_file_handle.name
                    self._ca_verify_file_handle.close()

                # Other (POSIX/Unix/Linux/OSX)
                else:
                    self._ca_verify_file_handle = temp_ca_bundle()
                    self._ca_verify_file_handle.write(BYTE_CA_BUNDLE)
                    self._ca_verify_file_handle.flush()
                    self.ca_verify_filename = self._ca_verify_file_handle.name

                # register cleanup function for temp file.
                atexit.register(self._cleanup_ca_temp_file)

            else:  # False
                # disable warnings for SSL certs.
                urllib3.disable_warnings()
                self.ca_verify_filename = False
        else:  # Not True/False, assume path to file/dir for Requests
            self.ca_verify_filename = self.verify
        return