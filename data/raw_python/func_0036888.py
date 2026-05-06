def tokenize(self, docs):
        """
        The first pass consists of converting documents
        into "transactions" (sets of their tokens)
        and the initial frequency/support filtering.

        Then iterate until we close in on a final set.

        `docs` can be any iterator or generator so long as it yields lists.
        Each list represents a document (i.e. is a list of tokens).
        For example, it can be a list of lists of nouns and noun phrases if trying
        to identify aspects, where each list represents a sentence or document.

        `min_sup` defines the minimum frequency (as a ratio over the total) necessary to
        keep a candidate.
        """
        if self.min_sup < 1/len(docs):
            raise Exception('`min_sup` must be greater than or equal to `1/len(docs)`.')

        # First pass
        candidates = set()
        transactions = []

        # Use nouns and noun phrases.
        for doc in POSTokenizer().tokenize(docs):
            transaction = set(doc)
            candidates = candidates.union({(t,) for t in transaction})
            transactions.append(transaction)
        freq_set = filter_support(candidates, transactions, self.min_sup)

        # Iterate
        k = 2
        last_set = set()
        while freq_set != set():
            last_set = freq_set
            cands = generate_candidates(freq_set, k)
            freq_set = filter_support(cands, transactions, self.min_sup)
            k += 1

        # Map documents to their keywords.
        keywords = flatten(last_set)
        return prune([[kw for kw in keywords if kw in doc] for doc in docs])