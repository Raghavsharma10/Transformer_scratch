def reparse(self):
        '''Reparse all children of this directory.

        This effectively rebuilds the tree below this node.

        This operation takes an unbounded time to complete; if there are a lot
        of objects registered below this directory's context, they will all
        need to be parsed.

        '''
        self._remove_all_children()
        self._parse_context(self._context, self.orb)