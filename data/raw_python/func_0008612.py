def value(self):
        """Return the :class:`Variable` or :class:`List` to watch.

        Returns ``None`` if it's a block watcher.

        """
        if self.kind == 'variable':
            return self.target.variables[self.block.args[0]]
        elif self.kind == 'list':
            return self.target.lists[self.block.args[0]]