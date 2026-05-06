def match_objects(self, set_a, set_b, time_a, time_b):
        """
        Match two sets of objects at particular times.

        Args:
            set_a: list of STObjects
            set_b: list of STObjects
            time_a: time at which set_a is being evaluated for matching
            time_b: time at which set_b is being evaluated for matching

        Returns:
            List of tuples containing (set_a index, set_b index) for each match
        """
        costs = self.cost_matrix(set_a, set_b, time_a, time_b) * 100
        min_row_costs = costs.min(axis=1)
        min_col_costs = costs.min(axis=0)
        good_rows = np.where(min_row_costs < 100)[0]
        good_cols = np.where(min_col_costs < 100)[0]
        assignments = []
        if len(good_rows) > 0 and len(good_cols) > 0:
            munk = Munkres()
            initial_assignments = munk.compute(costs[tuple(np.meshgrid(good_rows, good_cols, indexing='ij'))].tolist())
            initial_assignments = [(good_rows[x[0]], good_cols[x[1]]) for x in initial_assignments]
            for a in initial_assignments:
                if costs[a[0], a[1]] < 100:
                    assignments.append(a)
        return assignments