def train(self, category, text):
        """
        Trains a category with a sample of text

        :param category: the name of the category we want to train
        :type category: str
        :param text: the text we want to train the category with
        :type text: str
        """
        try:
            bayes_category = self.categories.get_category(category)
        except KeyError:
            bayes_category = self.categories.add_category(category)

        tokens = self.tokenizer(str(text))
        occurrence_counts = self.count_token_occurrences(tokens)

        for word, count in occurrence_counts.items():
            bayes_category.train_token(word, count)

        # Updating our per-category overall probabilities
        self.calculate_category_probability()