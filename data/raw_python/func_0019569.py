def to_pwm(self, precision=4, extra_str=""):
        """Return pwm as string.

        Parameters
        ----------
        precision : int, optional, default 4
            Floating-point precision.
        
        extra_str |: str, optional
            Extra text to include with motif id line.
        
        Returns
        -------
        motif_str : str
            Motif formatted in PWM format.
        """
        motif_id = self.id
        
        if extra_str:
            motif_id += "_%s" % extra_str

        if not self.pwm:
            self.pwm = [self.iupac_pwm[char]for char in self.consensus.upper()]

        return ">%s\n%s" % (
                motif_id, 
                self._pwm_to_str(precision)
            )