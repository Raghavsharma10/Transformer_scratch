def from_graph(cls, graph, length=0):
        """
        Creates a hypergraph (expanded graph) from a :class:`caspo.core.graph.Graph` object instance

        Parameters
        ----------
        graph : :class:`caspo.core.graph.Graph`
            The base interaction graph to be expanded

        length : int
            Maximum length for hyperedges source sets. If 0, use maximum possible in each case.

        Returns
        -------
        caspo.core.hypergraph.HyperGraph
            Created object instance
        """
        nodes = []
        hyper = []
        edges = defaultdict(list)
        j = 0

        for i, node in enumerate(graph.nodes_iter()):
            nodes.append(node)

            preds = graph.in_edges(node, data=True)
            l = len(preds)
            if length > 0:
                l = min(length, l)

            for literals in it.chain.from_iterable(it.combinations(preds, r+1) for r in xrange(l)):
                valid = defaultdict(int)
                for source, _, _ in literals:
                    valid[source] += 1

                if all(it.imap(lambda c: c == 1, valid.values())):
                    hyper.append(i)
                    for source, _, data in literals:
                        edges['hyper_idx'].append(j)
                        edges['name'].append(source)
                        edges['sign'].append(data['sign'])

                    j += 1

        nodes = pd.Series(nodes, name='name')
        hyper = pd.Series(hyper, name='node_idx')
        edges = pd.DataFrame(edges)

        return cls(nodes, hyper, edges)