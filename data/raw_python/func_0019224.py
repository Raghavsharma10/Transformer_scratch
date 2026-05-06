def norm_coefs(self):
        """Multiply all coefficients by the same factor, so that their sum
        becomes one."""
        sum_coefs = self.sum_coefs
        self.ar_coefs /= sum_coefs
        self.ma_coefs /= sum_coefs