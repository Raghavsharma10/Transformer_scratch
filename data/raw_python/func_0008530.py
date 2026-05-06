def _do_pnp(self, pnp, anchor=None):
        """ Attaches prepositional noun phrases.
            Identifies PNP's from either the PNP tag or the P-attachment tag.
            This does not determine the PP-anchor, it only groups words in a PNP chunk.
        """
        if anchor or pnp and pnp.endswith("PNP"):
            if anchor is not None:
                m = find(lambda x: x.startswith("P"), anchor)
            else:
                m = None
            if self.pnp \
             and pnp \
             and pnp != OUTSIDE \
             and pnp.startswith("B-") is False \
             and self.words[-2].pnp is not None:
                self.pnp[-1].append(self.words[-1])
            elif m is not None and m == self._attachment:
                self.pnp[-1].append(self.words[-1])
            else:
                ch = PNPChunk(self, [self.words[-1]], type="PNP")
                self.pnp.append(ch)                
            self._attachment = m