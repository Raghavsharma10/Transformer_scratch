def stripped_text(self):
        """The :attr:`text`, with spaces and inserts removed.

        Used by :class:`BlockType.get` to look up blocks.

        """
        return BaseBlockType._strip_text(
                self.text % tuple((i.default if i.shape == 'inline' else '%s')
                                  for i in self.inserts))