def _iter_descendants_levelorder(self, is_leaf_fn=None):
        """ Iterate over all desdecendant nodes."""
        tovisit = deque([self])
        while len(tovisit) > 0:
            node = tovisit.popleft()
            yield node
            if not is_leaf_fn or not is_leaf_fn(node):
                tovisit.extend(node.children)