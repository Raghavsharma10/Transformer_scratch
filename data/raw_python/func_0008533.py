def get(self, index, tag=LEMMA):
        """ Returns a tag for the word at the given index.
            The tag can be WORD, LEMMA, POS, CHUNK, PNP, RELATION, ROLE, ANCHOR or a custom word tag.
        """
        if tag == WORD:
            return self.words[index]
        if tag == LEMMA:
            return self.words[index].lemma
        if tag == POS:
            return self.words[index].type
        if tag == CHUNK:
            return self.words[index].chunk
        if tag == PNP:
            return self.words[index].pnp
        if tag == REL:
            ch = self.words[index].chunk; return ch and ch.relation
        if tag == ROLE:
            ch = self.words[index].chunk; return ch and ch.role
        if tag == ANCHOR:
            ch = self.words[index].pnp; return ch and ch.anchor
        if tag in self.words[index].custom_tags:
            return self.words[index].custom_tags[tag]
        return None