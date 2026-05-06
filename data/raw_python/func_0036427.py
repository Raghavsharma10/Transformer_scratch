def _core_semantics(self, lex_chains, concept_weights):
        """
        Returns the n representative lexical chains for a document.
        """
        chain_scores = [self._score_chain(lex_chain, adj_submat, concept_weights) for lex_chain, adj_submat in lex_chains]
        scored_chains = zip(lex_chains, chain_scores)
        scored_chains = sorted(scored_chains, key=lambda x: x[1], reverse=True)

        thresh = (self.alpha/len(lex_chains)) * sum(chain_scores)
        return [chain for (chain, adj_mat), score in scored_chains if score >= thresh][:self.n_chains]