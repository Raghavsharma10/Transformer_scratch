def score(self, text):
        """
        Scores a sample of text

        :param text: sample text to score
        :type text: str
        :return: dict of scores per category
        :rtype: dict
        """
        occurs = self.count_token_occurrences(self.tokenizer(text))
        scores = {}
        for category in self.categories.get_categories().keys():
            scores[category] = 0

        categories = self.categories.get_categories().items()

        for word, count in occurs.items():
            token_scores = {}

            # Adding up individual token scores
            for category, bayes_category in categories:
                token_scores[category] = \
                    float(bayes_category.get_token_count(word))

            # We use this to get token-in-category probabilities
            token_tally = sum(token_scores.values())

            # If this token isn't found anywhere its probability is 0
            if token_tally == 0.0:
                continue

            # Calculating bayes probabiltity for this token
            # http://en.wikipedia.org/wiki/Naive_Bayes_spam_filtering
            for category, token_score in token_scores.items():
                # Bayes probability * the number of occurances of this token
                scores[category] += count * \
                    self.calculate_bayesian_probability(
                        category,
                        token_score,
                        token_tally
                    )

        # Removing empty categories from the results
        final_scores = {}
        for category, score in scores.items():
            if score > 0:
                final_scores[category] = score

        return final_scores