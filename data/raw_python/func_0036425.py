def _semsim(self, c1, c2):
        """
        Computes the semantic similarity between two concepts.

        The semantic similarity is a combination of two sem sims:

            1. An "explicit" sem sim metric, that is, one which is directly
            encoded in the WordNet graph. Here it is just Wu-Palmer similarity.

            2. An "implicit" sem sim metric. See `_imp_semsim`.

        Note we can't use the NLTK Wu-Palmer similarity implementation because we need to
        incorporate the implicit sem sim, but it's fairly straightforward --
        leaning on <http://www.nltk.org/_modules/nltk/corpus/reader/wordnet.html#Synset.wup_similarity>,
        see that for more info. Though...the formula in the paper includes an extra term in the denominator,
        which is wrong, so we leave it out.
        """
        if c1 == c2:
            return 1.

        if (c1, c2) in self.concept_sims:
            return self.concept_sims[(c1, c2)]

        elif (c2, c1) in self.concept_sims:
            return self.concept_sims[(c2, c1)]

        else:
            need_root = c1._needs_root()
            subsumers = c1.lowest_common_hypernyms(c2, simulate_root=need_root)

            if not subsumers:
                # For relationships not in WordNet, fallback on just implicit semsim.
                return self._imp_semsim(c1, c2)

            subsumer = subsumers[0]
            depth = subsumer.max_depth() + 1
            len1 = c1.shortest_path_distance(subsumer, simulate_root=need_root)
            len2 = c2.shortest_path_distance(subsumer, simulate_root=need_root)

            if len1 is None or len2 is None:
                # See above
                return self._imp_semsim(c1, c2)

            len1 += depth
            len2 += depth

            imp_score = self._imp_semsim(c1, c2)

            sim = (2.*depth + imp_score)/(len1 + len2 + imp_score)
            self.concept_sims[(c1, c2)] = sim
            return sim