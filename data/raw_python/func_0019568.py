def _pwm_to_str(self, precision=4):
        """Return string representation of pwm.

        Parameters
        ----------
        precision : int, optional, default 4
            Floating-point precision.

        Returns
        -------
        pwm_string : str
        """
        if not self.pwm:
            return ""
        
        fmt = "{{:.{:d}f}}".format(precision)
        return "\n".join(
                ["\t".join([fmt.format(p) for p in row])
                for row in self.pwm]
                )