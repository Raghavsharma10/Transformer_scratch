def has_path(self, path):
        '''Check if a path exists below this node.

        @param path A list of path elements pointing to a node in the tree.
                    For example, ['/', 'localhost', 'dir.host']. The first
                    element in this path should be this node's name.
        @return True if the path points to a node in the tree below this node,
                or this node itself (for paths one element long). False
                otherwise.

        Example:
        >>> c1 = TreeNode(name='c1')
        >>> c2 = TreeNode(name='c2')
        >>> p = TreeNode(name='p', children={'c1':c1, 'c2':c2})
        >>> c1._parent = p
        >>> c2._parent = p
        >>> p.has_path(['p', 'c1'])
        True
        >>> p.has_path(['p', 'c3'])
        False
        '''
        with self._mutex:
            if path[0] == self._name:
                if len(path) == 1:
                    return True
                elif path[1] in self._children:
                    return self._children[path[1]].has_path(path[1:])
                else:
                    return False
            else:
                return False