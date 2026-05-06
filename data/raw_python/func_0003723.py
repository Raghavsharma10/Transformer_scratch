def add_hydrogens(self, formal_charges=None):
        """Returns a molecular graph where hydrogens are added explicitely

           When the bond order is unknown, it assumes bond order one. If the
           graph  has an attribute formal_charges, this routine will take it
           into account when counting the number of hydrogens to be added. The
           returned graph will also have a formal_charges attribute.

           This routine only adds hydrogen atoms for a limited set of atoms from
           the periodic system: B, C, N, O, F, Al, Si, P, S, Cl, Br.
        """

        new_edges = list(self.edges)
        counter = self.num_vertices
        for i in range(self.num_vertices):
            num_elec = self.numbers[i]
            if formal_charges is not None:
                num_elec -= int(formal_charges[i])
            if num_elec >= 5 and num_elec <= 9:
                num_hydrogen = num_elec - 10 + 8
            elif num_elec >= 13 and num_elec <= 17:
                num_hydrogen = num_elec - 18 + 8
            elif num_elec == 35:
                num_hydrogen = 1
            else:
                continue
            if num_hydrogen > 4:
                num_hydrogen = 8 - num_hydrogen
            for n in self.neighbors[i]:
                bo = self.orders[self.edge_index[frozenset([i, n])]]
                if bo <= 0:
                    bo = 1
                num_hydrogen -= int(bo)
            for j in range(num_hydrogen):
                new_edges.append((i, counter))
                counter += 1
        new_numbers = np.zeros(counter, int)
        new_numbers[:self.num_vertices] = self.numbers
        new_numbers[self.num_vertices:] = 1
        new_orders = np.zeros(len(new_edges), int)
        new_orders[:self.num_edges] = self.orders
        new_orders[self.num_edges:] = 1
        result = MolecularGraph(new_edges, new_numbers, new_orders)
        return result