def _do_chunk(self, type, role=None, relation=None, iob=None):
        """ Adds a new Chunk to the sentence, or adds the last word to the previous chunk.
            The word is attached to the previous chunk if both type and relation match,
            and if the word's chunk tag does not start with "B-" (i.e., iob != BEGIN).
            Punctuation marks (or other "O" chunk tags) are not chunked.
        """
        if (type is None or type == OUTSIDE) and \
           (role is None or role == OUTSIDE) and (relation is None or relation == OUTSIDE):
            return
        if iob != BEGIN \
         and self.chunks \
         and self.chunks[-1].type == type \
         and self._relation == (relation, role) \
         and self.words[-2].chunk is not None: # "one, two" => "one" & "two" different chunks.
            self.chunks[-1].append(self.words[-1])
        else:
            ch = Chunk(self, [self.words[-1]], type, role, relation)
            self.chunks.append(ch)
            self._relation = (relation, role)