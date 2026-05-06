def _assembled_out_file_name(self):
        """Checks file name is set for assembled output.
           Returns absolute path."""
        if self.Parameters['-s'].isOn():
            assembled_reads = self._absolute(str(self.Parameters['-s'].Value))
        else:
            raise ValueError(
                "No assembled-reads (flag -s) output path specified")
        return assembled_reads