def get_groups(self):
        """Return a list of groups of atom indexes

           Each atom in a group belongs to the same molecule or residue.
        """
        groups = []
        for a_index, m_index in enumerate(self.molecules):
            if m_index >= len(groups):
                groups.append([a_index])
            else:
                groups[m_index].append(a_index)
        return groups