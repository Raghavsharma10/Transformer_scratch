def _discarded_reads1_out_file_name(self):
        """Checks if file name is set for discarded reads1 output.
           Returns absolute path."""
        if self.Parameters['-3'].isOn():
            discarded_reads1 = self._absolute(str(self.Parameters['-3'].Value))
        else:
            raise ValueError(
                "No discarded-reads1 (flag -3) output path specified")
        return discarded_reads1