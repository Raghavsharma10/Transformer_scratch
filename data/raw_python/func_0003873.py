def get_next(self, label):
        """Get the next section with the given label"""
        while self._get_current_label() != label:
            self._skip_section()
        return self._read_section()