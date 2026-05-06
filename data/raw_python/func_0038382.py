def calculate_category_probability(self):
        """
        Caches the individual probabilities for each category
        """
        total_tally = 0.0
        probs = {}
        for category, bayes_category in \
                self.categories.get_categories().items():
            count = bayes_category.get_tally()
            total_tally += count
            probs[category] = count

        # Calculating the probability
        for category, count in probs.items():
            if total_tally > 0:
                probs[category] = float(count)/float(total_tally)
            else:
                probs[category] = 0.0

        for category, probability in probs.items():
            self.probabilities[category] = {
                # Probability that any given token is of this category
                'prc': probability,
                # Probability that any given token is not of this category
                'prnc': sum(probs.values()) - probability
            }