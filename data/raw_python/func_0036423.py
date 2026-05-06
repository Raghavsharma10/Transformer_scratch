def _disambiguate_pos(self, terms, pos):
        """
        Disambiguates a list of tokens of a given PoS.
        """
        # Map the terms to candidate concepts
        # Consider only the top 3 most common senses
        candidate_map = {term: wn.synsets(term, pos=pos)[:3] for term in terms}

        # Filter to unique concepts
        concepts = set(c for cons in candidate_map.values() for c in cons)

        # Back to list for consistent ordering
        concepts = list(concepts)
        sim_mat = self._similarity_matrix(concepts)

        # Final map of terms to their disambiguated concepts
        map = {}

        # This is terrible
        # For each term, select the candidate concept
        # which has the maximum aggregate similarity score against
        # all other candidate concepts of all other terms sharing the same PoS
        for term, cons in candidate_map.items():
            # Some words may not be in WordNet
            # and thus have no candidate concepts, so skip
            if not cons:
                continue
            scores = []
            for con in cons:
                i = concepts.index(con)
                scores_ = []
                for term_, cons_ in candidate_map.items():
                    # Some words may not be in WordNet
                    # and thus have no candidate concepts, so skip
                    if term == term_ or not cons_:
                        continue
                    cons_idx = [concepts.index(c) for c in cons_]
                    top_sim = max(sim_mat[i,cons_idx])
                    scores_.append(top_sim)
                scores.append(sum(scores_))
            best_idx = np.argmax(scores)
            map[term] = cons[best_idx]

        return map