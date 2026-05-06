def _edit1(self, w):
        """ Returns a set of words with edit distance 1 from the given word.
        """
        # Of all spelling errors, 80% is covered by edit distance 1.
        # Edit distance 1 = one character deleted, swapped, replaced or inserted.
        split = [(w[:i], w[i:]) for i in range(len(w) + 1)]
        delete, transpose, replace, insert = (
            [a + b[1:] for a, b in split if b],
            [a + b[1] + b[0] + b[2:] for a, b in split if len(b) > 1],
            [a + c + b[1:] for a, b in split for c in Spelling.ALPHA if b],
            [a + c + b[0:] for a, b in split for c in Spelling.ALPHA]
        )
        return set(delete + transpose + replace + insert)