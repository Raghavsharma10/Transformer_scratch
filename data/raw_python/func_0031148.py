def remove(self, name, strategy=Strategy.promote):
        """
        Remove a node from the graph. Returns the set of nodes that were
        removed.

        If the node doesn't exist, an exception will be raised.

        name: The name of the node to remove.
        strategy: (Optional, Strategy.promote) What to do with children
            or removed nodes. The options are:

            orphan: remove the node from the child's set of parents.
            promote: replace the node with the the node's parents in the
                childs set of parents.
            remove: recursively remove all children of the node.
        """
        removed = set()

        stack = [name]

        while stack:
            current = stack.pop()
            node = self._nodes.pop(current)

            if strategy == Strategy.remove:
                for child_name in node.children:
                    child_node = self._nodes[child_name]

                    child_node.parents.remove(current)

                    stack.append(child_name)
            else:
                for child_name in node.children:
                    child_node = self._nodes[child_name]

                    child_node.parents.remove(current)

                    if strategy == Strategy.promote:
                        for parent_name in node.parents:
                            parent_node = self._nodes[parent_name]

                            parent_node.children.add(child_name)
                            child_node.parents.add(parent_name)

                    if not child_node.parents:
                        self._roots.add(child_name)

            if current in self._stubs:
                self._stubs.remove(current)
            elif current in self._roots:
                self._roots.remove(current)
            else:  # stubs and roots (by definition) don't have parents
                for parent_name in node.parents:
                    parent_node = self._nodes[parent_name]

                    parent_node.children.remove(current)

                    if parent_name in self._stubs and not parent_node.children:
                        stack.append(parent_name)

            removed.add(current)

        return removed