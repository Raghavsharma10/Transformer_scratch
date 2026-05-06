def _lexical_chains(self, doc, term_concept_map):
        """
        Builds lexical chains, as an adjacency matrix,
        using a disambiguated term-concept map.
        """
        concepts = list({c for c in term_concept_map.values()})

        # Build an adjacency matrix for the graph
        # Using the encoding:
        # 1 = identity/synonymy, 2 = hypernymy/hyponymy, 3 = meronymy, 0 = no edge
        n_cons = len(concepts)
        adj_mat = np.zeros((n_cons, n_cons))

        for i, c in enumerate(concepts):
            # TO DO can only do i >= j since the graph is undirected
            for j, c_ in enumerate(concepts):
                edge = 0
                if c == c_:
                    edge = 1
                # TO DO when should simulate root be True?
                elif c_ in c._shortest_hypernym_paths(simulate_root=False).keys():
                    edge = 2
                elif c in c_._shortest_hypernym_paths(simulate_root=False).keys():
                    edge = 2
                elif c_ in c.member_meronyms() + c.part_meronyms() + c.substance_meronyms():
                    edge = 3
                elif c in c_.member_meronyms() + c_.part_meronyms() + c_.substance_meronyms():
                    edge = 3

                adj_mat[i,j] = edge

        # Group connected concepts by labels
        concept_labels = connected_components(adj_mat, directed=False)[1]
        lexical_chains = [([], []) for i in range(max(concept_labels) + 1)]
        for i, concept in enumerate(concepts):
            label = concept_labels[i]
            lexical_chains[label][0].append(concept)
            lexical_chains[label][1].append(i)

        # Return the lexical chains as (concept list, adjacency sub-matrix) tuples
        return [(chain, adj_mat[indices][:,indices]) for chain, indices in lexical_chains]