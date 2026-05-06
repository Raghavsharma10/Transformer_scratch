def _compute_radii(self):
        """Generate radii"""

        denom = pow(-np.log(self.grbf_lambda), 1.0/self.tau_vals)
        self.components_['radii'] = self.dF_vals/denom