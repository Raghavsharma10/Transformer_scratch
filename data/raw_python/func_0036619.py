def slaves(self):
        '''The list of slave managers of this manager, if any.

        This information can also be found by listing the children of this node
        that are of type @ref Manager.

        '''
        with self._mutex:
            if not self._slaves:
                self._slaves = [c for c in self.children if c.is_manager]
        return self._slaves