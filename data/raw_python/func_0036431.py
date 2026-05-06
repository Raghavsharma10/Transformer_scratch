def _weight_concepts(self, tokens, term_concept_map):
        """
        Calculates weights for concepts in a document.

        This is just the frequency of terms which map to a concept.
        """

        weights = {c: 0 for c in term_concept_map.values()}
        for t in tokens:
            # Skip terms that aren't one of the PoS we used
            if t not in term_concept_map:
                continue
            con = term_concept_map[t]
            weights[con] += 1

        # TO DO paper doesn't mention normalizing these weights...should we?
        return weights