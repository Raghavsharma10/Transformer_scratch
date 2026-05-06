def untrain(self, category, text):
        """
        Untrains a category with a sample of text

        :param category: the name of the category we want to train
        :type category: str
        :param text: the text we want to untrain the category with
        :type text: str
        """
        try:
            bayes_category = self.categories.get_category(category)
        except KeyError:
            return

        tokens = self.tokenizer(str(text))
        occurance_counts = self.count_token_occurrences(tokens)

        for word, count in occurance_counts.items():
            bayes_category.untrain_token(word, count)

        # Updating our per-category overall probabilities
        self.calculate_category_probability()