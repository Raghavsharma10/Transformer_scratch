def _description(self, concept):
        """
        Returns a "description" of a concept,
        as defined in the paper.

        The paper describes the description as a string,
        so this is a slight modification where we instead represent
        the definition as a list of tokens.
        """
        if concept not in self.descriptions:
            lemmas = concept.lemma_names()
            gloss = self._gloss(concept)
            glosses = [self._gloss(rel) for rel in self._related(concept)]
            raw_desc = ' '.join(lemmas + [gloss] + glosses)
            desc = [w for w in raw_desc.split() if w not in stops]
            self.descriptions[concept] = desc
        return self.descriptions[concept]