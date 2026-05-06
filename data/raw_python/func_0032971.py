def _check_status(self, ibsta):
        """Checks ibsta value."""
        if ibsta & 0x4000:
            raise LinuxGpib.Timeout()
        elif ibsta & 0x8000:
            raise LinuxGpib.Error(self.error_status)