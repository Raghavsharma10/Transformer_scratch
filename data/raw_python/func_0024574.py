def reset_query_marks(self):
        """
        set or reset hyb and neighbors marks to atoms.
        """
        for i, atom in self.atoms():
            neighbors = 0
            hybridization = 1
            p_neighbors = 0
            p_hybridization = 1
            # hyb 1- sp3; 2- sp2; 3- sp1; 4- aromatic
            for j, bond in self._adj[i].items():
                isnth = self._node[j].element != 'H'

                order = bond.order
                if order:
                    if isnth:
                        neighbors += 1
                    if hybridization not in (3, 4):
                        if order == 4:
                            hybridization = 4
                        elif order == 3:
                            hybridization = 3
                        elif order == 2:
                            if hybridization == 2:
                                hybridization = 3
                            else:
                                hybridization = 2
                order = bond.p_order
                if order:
                    if isnth:
                        p_neighbors += 1
                    if p_hybridization not in (3, 4):
                        if order == 4:
                            p_hybridization = 4
                        elif order == 3:
                            p_hybridization = 3
                        elif order == 2:
                            if p_hybridization == 2:
                                p_hybridization = 3
                            else:
                                p_hybridization = 2

            atom._reactant._neighbors = neighbors
            atom._reactant._hybridization = hybridization
            atom._product._neighbors = p_neighbors
            atom._product._hybridization = p_hybridization
            atom.__dict__.clear()  # flush cache
        self.flush_cache()