def reset_query_marks(self):
        """
        set or reset hyb and neighbors marks to atoms.
        """
        for i, atom in self.atoms():
            neighbors = 0
            hybridization = 1
            # hybridization 1- sp3; 2- sp2; 3- sp1; 4- aromatic
            for j, bond in self._adj[i].items():
                if self._node[j].element != 'H':
                    neighbors += 1

                if hybridization in (3, 4):
                    continue
                order = bond.order
                if order == 4:
                    hybridization = 4
                elif order == 3:
                    hybridization = 3
                elif order == 2:
                    if hybridization == 2:
                        hybridization = 3
                    else:
                        hybridization = 2

            atom._neighbors = neighbors
            atom._hybridization = hybridization
        self.flush_cache()