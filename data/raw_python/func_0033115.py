def _unassembled_reads1_out_file_name(self):
        """Checks file name is set for reads1 output.
           Returns absolute path."""
        if self.Parameters['-1'].isOn():
            unassembled_reads1 = self._absolute(
                str(self.Parameters['-1'].Value))
        else:
            raise ValueError("No reads1 (flag: -1) output path specified")
        return unassembled_reads1