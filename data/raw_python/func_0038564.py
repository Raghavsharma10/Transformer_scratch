def add_category(self, name):
        """
        Adds a bayes category that we can later train

        :param name: name of the category
        :type name: str
        :return: the requested category
        :rtype: BayesCategory
        """
        category = BayesCategory(name)
        self.categories[name] = category
        return category