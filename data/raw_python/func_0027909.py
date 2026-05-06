def most_frequent_terms(self, depth):

        """
        Get the X most frequent terms in the text, and then probe down to get
        any other terms that have the same count as the last term.

        Args:
            depth (int): The number of terms.

        Returns:
            set: The set of frequent terms.
        """

        counts = self.term_counts()

        # Get the top X terms and the instance count of the last word.
        top_terms = set(list(counts.keys())[:depth])
        end_count = list(counts.values())[:depth][-1]

        # Merge in all other words with that appear that number of times, so
        # that we don't truncate the last bucket - eg, half of the words that
        # appear 5 times, but not the other half.

        bucket = self.term_count_buckets()[end_count]
        return top_terms.union(set(bucket))