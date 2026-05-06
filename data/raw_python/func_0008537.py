def constituents(self, pnp=False):
        """ Returns an in-order list of mixed Chunk and Word objects.
            With pnp=True, also contains PNPChunk objects whenever possible.
        """
        a = []
        for word in self.words:
            if pnp and word.pnp is not None:
                if len(a) == 0 or a[-1] != word.pnp:
                    a.append(word.pnp)
            elif word.chunk is not None:
                if len(a) == 0 or a[-1] != word.chunk:
                    a.append(word.chunk)
            else:
                a.append(word)
        return a