def getElementById(self, _id, root='root'):
        '''
            getElementById - Searches and returns the first (should only be one) element with the given ID.
               
                @param id <str> - A string of the id attribute.
                @param root <AdvancedTag/'root'> - Search starting at a specific node, if provided. if string 'root' [default], the root of the parsed tree will be used.
        '''
        (root, isFromRoot) = self._handleRootArg(root)

        if isFromRoot is True and root.id == _id:
            return root

        getElementById = self.getElementById
        for child in root.children:

            if child.getAttribute('id') == _id:
                return child

            potential = getElementById(_id, child)
            if potential is not None:
                return potential

        return None