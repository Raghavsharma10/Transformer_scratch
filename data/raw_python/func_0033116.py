def _unassembled_reads2_out_file_name(self):
        """Checks if file name is set for reads2 output.
           Returns absolute path."""
        if self.Parameters['-2'].isOn():
            unassembled_reads2 = self._absolute(
                str(self.Parameters['-2'].Value))
        else:
            raise ValueError("No reads2 (flag -2) output path specified")
        return unassembled_reads2