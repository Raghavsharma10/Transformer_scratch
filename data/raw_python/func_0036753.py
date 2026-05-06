def get_node(self, path):
        '''Get a child node of this node, or this node, based on a path.

        @param path A list of path elements pointing to a node in the tree.
                    For example, ['/', 'localhost', 'dir.host']. The first
                    element in this path should be this node's name.
        @return The node pointed to by @ref path, or None if the path does not
                point to a node in the tree below this node.

        Example:
        >>> c1 = TreeNode(name='c1')
        >>> c2 = TreeNode(name='c2')
        >>> p = TreeNode(name='p', children={'c1':c1, 'c2':c2})
        >>> c1._parent = p
        >>> c2._parent = p
        >>> p.get_node(['p', 'c1']) == c1
        True
        >>> p.get_node(['p', 'c2']) == c2
        True
        '''
        with self._mutex:
            if path[0] == self._name:
                if len(path) == 1:
                    return self
                elif path[1] in self._children:
                    return self._children[path[1]].get_node(path[1:])
                else:
                    return None
            else:
                return None