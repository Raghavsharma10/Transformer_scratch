def _update_Prxy(self):
        """Update `Prxy` using current `Frxy` and `Qxy`."""
        self.Prxy = self.Frxy * self.Qxy
        _fill_diagonals(self.Prxy, self._diag_indices)