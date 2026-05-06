def tally(self, category):
        """
        Gets the tally for a requested category

        :param category: The category we want a tally for
        :type category: str
        :return: tally for a given category
        :rtype: int
        """
        try:
            bayes_category = self.categories.get_category(category)
        except KeyError:
            return 0

        return bayes_category.get_tally()