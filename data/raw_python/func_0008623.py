def copy(self):
        """Return a new Block instance with the same attributes."""
        args = []
        for arg in self.args:
            if isinstance(arg, Block):
                arg = arg.copy()
            elif isinstance(arg, list):
                arg = [b.copy() for b in arg]
            args.append(arg)
        return Block(self.type, *args)