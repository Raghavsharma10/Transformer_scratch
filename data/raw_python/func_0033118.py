def _discarded_reads2_out_file_name(self):
        """Checks if file name is set for discarded reads2 output.
           Returns absolute path."""
        if self.Parameters['-4'].isOn():
            discarded_reads2 = self._absolute(str(self.Parameters['-4'].Value))
        else:
            raise ValueError(
                "No discarded-reads2 (flag -4) output path specified")
        return discarded_reads2