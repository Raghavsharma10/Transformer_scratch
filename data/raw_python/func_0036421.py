def _process_doc(self, doc):
        """
        Applies DCS to a document to extract its core concepts and their weights.
        """
        # Prep
        doc = doc.lower()
        tagged_tokens = [(t, penn_to_wordnet(t.tag_)) for t in spacy(doc, tag=True, parse=False, entity=False)]
        tokens = [t for t, tag in tagged_tokens]
        term_concept_map = self._disambiguate_doc(tagged_tokens)
        concept_weights = self._weight_concepts(tokens, term_concept_map)

        # Compute core semantics
        lexical_chains = self._lexical_chains(doc, term_concept_map)
        core_semantics = self._core_semantics(lexical_chains, concept_weights)
        core_concepts = [c for chain in core_semantics for c in chain]

        return [(con, concept_weights[con]) for con in core_concepts]