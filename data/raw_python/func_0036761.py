def root(self):
        '''The root node of the tree this node is in.'''
        with self._mutex:
            if self._parent:
                return self._parent.root
            else:
                return self