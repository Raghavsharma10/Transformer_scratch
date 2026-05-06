def _disambiguate_doc(self, tagged_tokens):
        """
        Takes a list of tagged tokens, representing a document,
        in the form:

            [(token, tag), ...]

        And returns a mapping of terms to their disambiguated concepts (synsets).
        """

        # Group tokens by PoS
        pos_groups = {pos: [] for pos in [wn.NOUN, wn.VERB, wn.ADJ, wn.ADV]}
        for tok, tag in tagged_tokens:
            if tag in pos_groups:
                pos_groups[tag].append(tok)

        #print(pos_groups)

        # Map of final term -> concept mappings
        map = {}
        for tag, toks in pos_groups.items():
            map.update(self._disambiguate_pos(toks, tag))

        #nice_map = {k: map[k].lemma_names() for k in map.keys()}
        #print(json.dumps(nice_map, indent=4, sort_keys=True))

        return map