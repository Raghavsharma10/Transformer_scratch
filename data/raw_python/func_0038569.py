def compress(self, setup):
        """
        Returns the compressed graph according to the given experimental setup

        Parameters
        ----------
        setup : :class:`caspo.core.setup.Setup`
            Experimental setup used to compress the graph

        Returns
        -------
        caspo.core.graph.Graph
            Compressed graph
        """
        designated = set(setup.nodes)
        zipped = self.copy()

        marked = [(n, d) for n, d in self.nodes(data=True) if n not in designated and not d.get('compressed', False)]
        while marked:
            for node, _ in sorted(marked):
                backward = zipped.predecessors(node)
                forward = zipped.successors(node)

                if not backward or (len(backward) == 1 and not backward[0] in forward):
                    self.__merge_source_targets(node, zipped)

                elif not forward or (len(forward) == 1 and not forward[0] in backward):
                    self.__merge_target_sources(node, zipped)

                else:
                    designated.add(node)

            marked = [(n, d) for n, d in self.nodes(data=True) if n not in designated and not d.get('compressed', False)]

        not_compressed = [(n, d) for n, d in zipped.nodes(data=True) if not d.get('compressed', False)]
        return zipped.subgraph([n for n, _ in not_compressed])