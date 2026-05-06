def full_path_str(self):
        '''The full path of this node as a string.'''
        with self._mutex:
            if self._parent:
                if self._parent._name == '/':
                    return self._parent.full_path_str + self._name
                else:
                    return self._parent.full_path_str + '/' + self._name
            else:
                return self._name