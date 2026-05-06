def _init_orient(self):
        """Retrieve the quadrature points and weights if needed.
        """
        if self.orient == orientation.orient_averaged_fixed:
            (self.beta_p, self.beta_w) = quadrature.get_points_and_weights(
                self.or_pdf, 0, 180, self.n_beta)
        self._set_orient_signature()