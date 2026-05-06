def _get_current_label(self):
        """Get the label from the last line read"""
        if len(self._last) == 0:
            raise StopIteration
        return self._last[:self._last.find(":")]