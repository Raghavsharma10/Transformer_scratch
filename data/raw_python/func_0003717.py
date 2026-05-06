def from_geometry(cls, molecule, do_orders=False, scaling=1.0):
        """Construct a MolecularGraph object based on interatomic distances

           All short distances are computed with the binning module and compared
           with a database of bond lengths. Based on this comparison, bonded
           atoms are detected.

           Before marking a pair of atoms A and B as bonded, it is also checked
           that there is no third atom C somewhat between A and B.
           When an atom C exists that is closer to B (than A) and the angle
           A-B-C is less than 45 degrees, atoms A and B are not bonded.
           Similarly if C is closer to A (than B) and the angle B-A-C is less
           then 45 degrees, A and B are not connected.

           Argument:
            | ``molecule``  --  The molecule to derive the graph from

           Optional arguments:
            | ``do_orders``  --  set to True to estimate the bond order
            | ``scaling``  --  scale the threshold for the connectivity. increase
                               this to 1.5 in case of transition states when a
                               fully connected topology is required.
        """
        from molmod.bonds import bonds

        unit_cell = molecule.unit_cell
        pair_search = PairSearchIntra(
            molecule.coordinates,
            bonds.max_length*bonds.bond_tolerance*scaling,
            unit_cell
        )

        orders = []
        lengths = []
        edges = []

        for i0, i1, delta, distance in pair_search:
            bond_order = bonds.bonded(molecule.numbers[i0], molecule.numbers[i1], distance/scaling)
            if bond_order is not None:
                if do_orders:
                    orders.append(bond_order)
                lengths.append(distance)
                edges.append((i0,i1))

        if do_orders:
            result = cls(edges, molecule.numbers, orders, symbols=molecule.symbols)
        else:
            result = cls(edges, molecule.numbers, symbols=molecule.symbols)

        # run a check on all neighbors. if two bonds point in a direction that
        # differs only by 45 deg. the longest of the two is discarded. the
        # double loop over the neighbors is done such that the longest bonds
        # are eliminated first
        slated_for_removal = set([])
        threshold = 0.5**0.5
        for c, ns in result.neighbors.items():
            lengths_ns = []
            for n in ns:
                delta = molecule.coordinates[n] - molecule.coordinates[c]
                if unit_cell is not None:
                    delta = unit_cell.shortest_vector(delta)
                length = np.linalg.norm(delta)
                lengths_ns.append([length, delta, n])
            lengths_ns.sort(reverse=True, key=(lambda r: r[0]))
            for i0, (length0, delta0, n0) in enumerate(lengths_ns):
                for i1, (length1, delta1, n1) in enumerate(lengths_ns[:i0]):
                    if length1 == 0.0:
                        continue
                    cosine = np.dot(delta0, delta1)/length0/length1
                    if cosine > threshold:
                        # length1 > length0
                        slated_for_removal.add((c,n1))
                        lengths_ns[i1][0] = 0.0
        # construct a mask
        mask = np.ones(len(edges), bool)
        for i0, i1 in slated_for_removal:
            edge_index = result.edge_index.get(frozenset([i0,i1]))
            if edge_index is None:
                raise ValueError('Could not find edge that has to be removed: %i %i' % (i0, i1))
            mask[edge_index] = False
        # actual removal
        edges = [edges[i] for i in range(len(edges)) if mask[i]]
        if do_orders:
            bond_order = [bond_order[i] for i in range(len(bond_order)) if mask[i]]
            result = cls(edges, molecule.numbers, orders)
        else:
            result = cls(edges, molecule.numbers)

        lengths = [lengths[i] for i in range(len(lengths)) if mask[i]]
        result.bond_lengths = np.array(lengths)

        return result