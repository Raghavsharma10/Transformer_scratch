def ancestor_of(self, name, ancestor, visited=None):
        """
        Check whether a node has another node as an ancestor.

        name: The name of the node being checked.
        ancestor: The name of the (possible) ancestor node.
        visited: (optional, None) If given, a set of nodes that have
            already been traversed. NOTE: The set will be updated with
            any new nodes that are visited.

        NOTE: If node doesn't exist, the method will return False.
        """
        if visited is None:
            visited = set()

        node = self._nodes.get(name)

        if node is None or name not in self._nodes:
            return False

        stack = list(node.parents)

        while stack:
            current = stack.pop()

            if current == ancestor:
                return True

            if current not in visited:
                visited.add(current)

                node = self._nodes.get(current)
                if node is not None:
                    stack.extend(node.parents)

        return False