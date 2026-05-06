def pwm_min_score(self):
        """Return the minimum PWM score.

        Returns
        -------
        score : float
            Minimum PWM score.
        """
        if self.min_score is None:
            score = 0
            for row in self.pwm:
                score += log(min(row) / 0.25 + 0.01)
            self.min_score = score
        
        return self.min_score