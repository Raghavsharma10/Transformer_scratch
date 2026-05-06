def pluralize(self, measure, singular, plural):
        """ Returns a string that contains the measure (amount) and its plural
        or singular form depending on the amount.

        Parameters:
            :param measure: Amount, value, always a numerical value
            :param singular: The singular form of the chosen word
            :param plural: The plural form of the chosen word

        Returns:
            String
        """
        if measure == 1:
            return "{} {}".format(measure, singular)
        else:
            return "{} {}".format(measure, plural)