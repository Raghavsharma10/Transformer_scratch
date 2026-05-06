def confidence(self):
        """
        Returns a tuple (chi squared, confident) of the experiment. Confident
        is simply a boolean specifying whether we're > 95%% sure that the
        results are statistically significant.
        """

        choices = self.choices

        # Get the chi-squared between the top two choices, if more than two choices exist
        if len(choices) >= 2:
            csq = chi_squared(*choices)
            confident = is_confident(csq, len(choices)) if len(choices) <= 10 else None
        else:
            csq = None
            confident = False

        return (csq, confident)