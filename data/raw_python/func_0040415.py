def apply_noise(self, noise_weights=None, uniform_amount=0.1):
        """
        Add noise to every link in the network.

        Can use either a ``uniform_amount`` or a ``noise_weight`` weight
        profile. If ``noise_weight`` is set, ``uniform_amount`` will be
        ignored.

        Args:
            noise_weights (list): a list of weight tuples
                of form ``(float, float)`` corresponding to
                ``(amount, weight)`` describing the noise to be
                added to each link in the graph
            uniform_amount (float): the maximum amount of uniform noise
                to be applied if ``noise_weights`` is not set

        Returns: None

        Example:
            >>> from blur.markov.node import Node
            >>> node_1 = Node('One')
            >>> node_2 = Node('Two')
            >>> node_1.add_link(node_1, 3)
            >>> node_1.add_link(node_2, 5)
            >>> node_2.add_link(node_1, 1)
            >>> graph = Graph([node_1, node_2])
            >>> for link in graph.node_list[0].link_list:
            ...     print('{} {}'.format(link.target.value, link.weight))
            One 3
            Two 5
            >>> graph.apply_noise()
            >>> for link in graph.node_list[0].link_list:
            ...     print('{} {}'.format(
            ...         link.target.value, link.weight))       # doctest: +SKIP
            One 3.154
            Two 5.321
        """
        # Main node loop
        for node in self.node_list:
            for link in node.link_list:
                if noise_weights is not None:
                    noise_amount = round(weighted_rand(noise_weights), 3)
                else:
                    noise_amount = round(random.uniform(
                        0, link.weight * uniform_amount), 3)
                link.weight += noise_amount