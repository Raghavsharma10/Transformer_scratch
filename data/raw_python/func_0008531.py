def _do_anchor(self, anchor):
        """ Collects preposition anchors and attachments in a dictionary.
            Once the dictionary has an entry for both the anchor and the attachment, they are linked.
        """
        if anchor:
            for x in anchor.split("-"):
                A, P = None, None
                if x.startswith("A") and len(self.chunks) > 0: # anchor
                    A, P = x, x.replace("A","P")
                    self._anchors[A] = self.chunks[-1]
                if x.startswith("P") and len(self.pnp) > 0:    # attachment (PNP)
                    A, P = x.replace("P","A"), x
                    self._anchors[P] = self.pnp[-1]
                if A in self._anchors and P in self._anchors and not self._anchors[P].anchor:
                    pnp = self._anchors[P]
                    pnp.anchor = self._anchors[A]
                    pnp.anchor.attachments.append(pnp)