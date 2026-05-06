def is_end(self, char):
        """
        Return `True` if this `char` is part of this node's letter set,
        `False` otherwise.
        """
        char = char.lower()

        return bool(cgaddag.gdg_is_end(self.gdg, self.node, char.encode("ascii")))