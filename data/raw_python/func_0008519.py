def anchor_id(self):
        """ Yields the anchor tag as parsed from the original token.
            Chunks that are anchors have a tag with an "A" prefix (e.g., "A1").
            Chunks that are PNP attachmens (or chunks inside a PNP) have "P" (e.g., "P1").
            Chunks inside a PNP can be both anchor and attachment (e.g., "P1-A2"),
            as in: "clawed/A1 at/P1 mice/P1-A2 in/P2 the/P2 wall/P2"
        """
        id = ""
        f = lambda ch: filter(lambda k: self.sentence._anchors[k] == ch, self.sentence._anchors)
        if self.pnp and self.pnp.anchor:
            id += "-" + "-".join(f(self.pnp))
        if self.anchor:
            id += "-" + "-".join(f(self))
        if self.attachments:
            id += "-" + "-".join(f(self))
        return id.strip("-") or None