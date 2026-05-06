def add_nodes(self, nodes):
        """
        Add a given node or list of nodes to self.node_list.

        Args:
            node (Node or list[Node]): the node or list of nodes to add
                to the graph

        Returns: None

        Examples:

        Adding one node: ::

            >>> from blur.markov.node import Node
            >>> graph = Graph()
            >>> node_1 = Node('One')
            >>> graph.add_nodes(node_1)
            >>> print([node.value for node in graph.node_list])
            ['One']

        Adding multiple nodes at a time in a list: ::

            >>> from blur.markov.node import Node
            >>> graph = Graph()
            >>> node_1 = Node('One')
            >>> node_2 = Node('Two')
            >>> graph.add_nodes([node_1, node_2])
            >>> print([node.value for node in graph.node_list])
            ['One', 'Two']
        """
        # Generalize nodes to a list
        if not isinstance(nodes, list):
            add_list = [nodes]
        else:
            add_list = nodes
        self.node_list.extend(add_list)