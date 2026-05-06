def term_count_buckets(self):

        """
        Returns:
            dict: A dictionary that maps occurrence counts to the terms that
            appear that many times in the text.
        """

        buckets = {}
        for term, count in self.term_counts().items():
            if count in buckets: buckets[count].append(term)
            else: buckets[count] = [term]

        return buckets