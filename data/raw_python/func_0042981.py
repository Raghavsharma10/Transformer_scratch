def values(self):
        """
        TRY NOT TO USE THIS, IT IS SLOW
        """
        matrix = self.data.values()[0]  # CANONICAL REPRESENTATIVE
        if matrix.num == 0:
            return
        e_names = self.edges.name
        s_names = self.select.name
        parts = [e.domain.partitions.value if e.domain.primitive else e.domain.partitions for e in self.edges]
        for c in matrix._all_combos():
            try:
                output = {n: parts[i][c[i]] for i, n in enumerate(e_names)}
            except Exception as e:
                Log.error("problem", cause=e)
            for s in s_names:
                output[s] = self.data[s][c]
            yield wrap(output)