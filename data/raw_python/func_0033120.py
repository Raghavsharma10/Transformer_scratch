def _pretty_alignment_out_file_name(self):
        """Checks file name is set for pretty alignment output.
           Returns absolute path."""
        if self.Parameters['-E'].isOn():
            pretty_alignment = self._absolute(str(self.Parameters['-E'].Value))
        else:
            raise ValueError(
                "No pretty-=alignment (flag -E) output path specified")
        return pretty_alignment