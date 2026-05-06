def _get_output_path(self):
        """Checks if a base file label / path is set. Returns absolute path."""
        if self.Parameters['-o'].isOn():
            output_path = self._absolute(str(self.Parameters['-o'].Value))
        else:
            raise ValueError("No output path specified.")
        return output_path