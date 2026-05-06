def text(self):
        """The text displayed on the block.

        String containing ``"%s"`` in place of inserts.

        eg. ``'say %s for %s secs'``

        """
        parts = [("%s" if isinstance(p, Insert) else p) for p in self.parts]
        parts = [("%%" if p == "%" else p) for p in parts] # escape percent
        return "".join(parts)