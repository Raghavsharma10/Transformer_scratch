def calculate_bayesian_probability(self, cat, token_score, token_tally):
        """
        Calculates the bayesian probability for a given token/category

        :param cat: The category we're scoring for this token
        :type cat: str
        :param token_score: The tally of this token for this category
        :type token_score: float
        :param token_tally: The tally total for this token from all categories
        :type token_tally: float
        :return: bayesian probability
        :rtype: float
        """
        # P that any given token IS in this category
        prc = self.probabilities[cat]['prc']
        # P that any given token is NOT in this category
        prnc = self.probabilities[cat]['prnc']
        # P that this token is NOT of this category
        prtnc = (token_tally - token_score) / token_tally
        # P that this token IS of this category
        prtc = token_score / token_tally

        # Assembling the parts of the bayes equation
        numerator = (prtc * prc)
        denominator = (numerator + (prtnc * prnc))

        # Returning the calculated bayes probability unless the denom. is 0
        return numerator / denominator if denominator != 0.0 else 0.0