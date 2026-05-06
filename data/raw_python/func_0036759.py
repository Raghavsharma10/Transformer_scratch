def nameserver(self):
        '''The name server of the node (i.e. its top-most parent below /).'''
        with self._mutex:
            if not self._parent:
                # The root node does not have a name server
                return None
            elif self._parent.name == '/':
                return self
            else:
                return self._parent.nameserver