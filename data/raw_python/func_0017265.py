def filter_clades(self):
        "Remove conflicting clades and those < cutoff to get majority rule"
        passed = []
        carrs = np.array([list(i[0]) for i in self.clade_counts], dtype=int)
        freqs = np.array([i[1] for i in self.clade_counts])

        for idx in range(carrs.shape[0]):
            conflict = False
            if freqs[idx] < self.cutoff:
                continue
            
            for pidx in passed:
                intersect = np.max(carrs[idx] + carrs[pidx]) > 1
                
                # is either one a subset of the other?
                subset_test0 = np.all(carrs[idx] - carrs[pidx] >= 0)
                subset_test1 = np.all(carrs[pidx] - carrs[idx] >= 0)
                if intersect:
                    if (not subset_test0) and (not subset_test1):
                        conflict = True

            if not conflict:
                passed.append(idx)

        rclades = []
        for idx in passed:
            rclades.append((carrs[idx], freqs[idx]))
        self.fclade_counts = rclades