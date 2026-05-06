def trim(self, edge_ic_cutoff=0.4):
        """Trim positions with an information content lower than the threshold.

        The default threshold is set to 0.4. The Motif will be changed in-place.

        Parameters
        ----------
        edge_ic_cutoff : float, optional
            Information content threshold. All motif positions at the flanks 
            with an information content lower thab this will be removed.

        Returns
        -------
        m : Motif instance
        """
        pwm = self.pwm[:]
        while len(pwm) > 0 and self.ic_pos(pwm[0]) < edge_ic_cutoff:
            pwm = pwm[1:]
            self.pwm = self.pwm[1:]
            self.pfm = self.pfm[1:]
        while len(pwm) > 0 and self.ic_pos(pwm[-1]) < edge_ic_cutoff:
            pwm = pwm[:-1]
            self.pwm = self.pwm[:-1]
            self.pfm = self.pfm[:-1]
        
        self.consensus = None 
        self.min_score = None
        self.max_score = None
        self.wiggled_pwm = None
        
        return self