def add(self, name, parents=None):
        """
        add a node to the graph.

        Raises an exception if the node cannot be added (i.e., if a node
        that name already exists, or if it would create a cycle.

        NOTE: A node can be added before its parents are added.

        name: The name of the node to add to the graph. Name can be any
            unique Hashable value.
        parents: (optional, None) The name of the nodes parents.
        """
        if not isinstance(name, Hashable):
            raise TypeError(name)

        parents = set(parents or ())
        is_stub = False

        if name in self._nodes:
            if name in self._stubs:
                node = Node(name, self._nodes[name].children, parents)
                is_stub = True
            else:
                raise ValueError(name)
        else:
            node = Node(name, set(), parents)

        # cycle detection
        visited = set()

        for parent in parents:
            if self.ancestor_of(parent, name, visited=visited):
                raise ValueError(parent)
            elif parent == name:
                raise ValueError(parent)

        # Node safe to add
        if is_stub:
            self._stubs.remove(name)

        if parents:
            for parent_name in parents:
                parent_node = self._nodes.get(parent_name)

                if parent_node is not None:
                    parent_node.children.add(name)
                else:  # add stub
                    self._nodes[parent_name] = Node(
                        name=parent_name,
                        children=set((name,)),
                        parents=frozenset(),
                    )
                    self._stubs.add(parent_name)
        else:
            self._roots.add(name)

        self._nodes[name] = node