def _score_chain(self, lexical_chain, adj_submat, concept_weights):
        """
        Computes the score for a lexical chain.
        """
        scores = []

        # Compute scores for concepts in the chain
        for i, c in enumerate(lexical_chain):
            score = concept_weights[c] * self.relation_weights[0]
            rel_scores = []
            for j, c_ in enumerate(lexical_chain):
                if adj_submat[i,j] == 2:
                    rel_scores.append(self.relation_weights[1] * concept_weights[c_])

                elif adj_submat[i,j] == 3:
                    rel_scores.append(self.relation_weights[2] * concept_weights[c_])

            scores.append(score + sum(rel_scores))

        # The chain's score is just the sum of its concepts' scores
        return sum(scores)