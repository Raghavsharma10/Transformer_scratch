def components(self):
        '''The list of components in this manager, if any.

        This information can also be found by listing the children of this node
        that are of type @ref Component. That method is more useful as it returns
        the tree entries for the components.

        '''
        with self._mutex:
            if not self._components:
                self._components = [c for c in self.children if c.is_component]
        return self._components