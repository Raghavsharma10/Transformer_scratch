def build(self, text, matrix, skim_depth=10, d_weights=False):

        """
        1. For each term in the passed matrix, score its KDE similarity with
        all other indexed terms.

        2. With the ordered stack of similarities in hand, skim off the top X
        pairs and add them as edges.

        Args:
            text (Text): The source text instance.
            matrix (Matrix): An indexed term matrix.
            skim_depth (int): The number of siblings for each term.
            d_weights (bool): If true, give "close" words low edge weights.
        """

        for anchor in bar(matrix.keys):

            n1 = text.unstem(anchor)

            # Heaviest pair scores:
            pairs = matrix.anchored_pairs(anchor).items()
            for term, weight in list(pairs)[:skim_depth]:

                # If edges represent distance, use the complement of the raw
                # score, so that similar words are connected by "short" edges.
                if d_weights: weight = 1-weight

                n2 = text.unstem(term)

                # NetworkX does not handle numpy types when writing graphml,
                # so we cast the weight to a regular float.
                self.graph.add_edge(n1, n2, weight=float(weight))