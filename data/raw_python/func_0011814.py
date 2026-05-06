def _check(self): # TODO: fix?
        """Do not check final CRC."""
        if self._returncode:
            rarfile.check_returncode(self, '')
        if self._remain != 0:
            raise rarfile.BadRarFile("Failed the read enough data")