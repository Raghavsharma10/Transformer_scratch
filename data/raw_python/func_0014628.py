def getRootNodes(self):
        '''
            getRootNodes - Gets all objects at the "root" (first level; no parent). Use this if you may have multiple roots (not children of <html>)
                Use this method to get objects, for example, in an AJAX request where <html> may not be your root.

                Note: If there are multiple root nodes (i.e. no <html> at the top), getRoot will return a special tag. This function automatically
                  handles that, and returns all root nodes.

                @return list<AdvancedTag> - A list of AdvancedTags which are at the root level of the tree.
        '''
        root = self.root
        if not root:
            return []
        if root.tagName == INVISIBLE_ROOT_TAG:
            return list(root.children)
        return [root]