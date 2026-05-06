def full_path(self):
        '''The full path of this node.'''
        with self._mutex:
            if self._parent:
                return self._parent.full_path + [self._name]
            else:
                return [self._name]