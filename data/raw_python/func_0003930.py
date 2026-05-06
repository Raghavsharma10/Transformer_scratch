def _update_cg(self):
        """Update the conjugate gradient"""
        beta = self._beta()
        # Automatic direction reset
        if beta < 0:
            self.direction = -self.gradient
            self.status = "SD"
        else:
            self.direction = self.direction * beta - self.gradient
            self.status = "CG"