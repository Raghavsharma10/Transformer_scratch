def kind(self):
        """The type of value to watch, based on :attr:`block`.

        One of ``variable``, ``list``, or ``block``.

        ``block`` watchers watch the value of a reporter block.

        """
        if self.block.type.has_command('readVariable'):
            return 'variable'
        elif self.block.type.has_command('contentsOfList:'):
            return 'list'
        else:
            return 'block'