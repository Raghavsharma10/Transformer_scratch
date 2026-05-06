def bonds(self):
        """
        iterate other all bonds
        """
        seen = set()
        for n, m_bond in self._adj.items():
            seen.add(n)
            for m, bond in m_bond.items():
                if m not in seen:
                    yield n, m, bond