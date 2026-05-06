def pwm_max_score(self):
        """Return the maximum PWM score.

        Returns
        -------
        score : float
            Maximum PWM score.
        """
        if self.max_score is None:
            score = 0
            for row in self.pwm:
                score += log(max(row) / 0.25 + 0.01)
            self.max_score = score
        
        return self.max_score