def standardize(self):
        """
        standardize functional groups

        :return: number of found groups
        """
        self.reset_query_marks()
        seen = set()
        total = 0
        for n, atom in self.atoms():
            if n in seen:
                continue
            for k, center in central.items():
                if center != atom:
                    continue
                shell = tuple((bond, self._node[m]) for m, bond in self._adj[n].items())
                for shell_query, shell_patch, atom_patch in query_patch[k]:
                    if shell_query != shell:
                        continue
                    total += 1
                    for attr_name, attr_value in atom_patch.items():
                        setattr(atom, attr_name, attr_value)
                    for (bond_patch, atom_patch), (bond, atom) in zip(shell_patch, shell):
                        bond.update(bond_patch)
                        for attr_name, attr_value in atom_patch.items():
                            setattr(atom, attr_name, attr_value)
                    seen.add(n)
                    seen.update(self._adj[n])
                    break
                else:
                    continue
                break
        if total:
            self.flush_cache()
        return total