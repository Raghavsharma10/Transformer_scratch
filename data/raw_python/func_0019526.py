def best_score(self, seqs, scan_rc=True, normalize=False):
        """
        give the score of the best match of each motif in each sequence
        returns an iterator of lists containing floats
        """
        self.set_threshold(threshold=0.0)
        if normalize and len(self.meanstd) == 0:
            self.set_meanstd()
            means = np.array([self.meanstd[m][0] for m in self.motif_ids])
            stds = np.array([self.meanstd[m][1] for m in self.motif_ids])

        for matches in self.scan(seqs, 1, scan_rc):
            scores = np.array([sorted(m, key=lambda x: x[0])[0][0] for m in matches if len(m) > 0])
            if normalize:
                scores = (scores - means) / stds
            yield scores