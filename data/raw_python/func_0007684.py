def update(self, indices):
        """Updates counts based on indices. The algorithm tracks the index change at i and
        update global counts for all indices beyond i with local counts tracked so far.
        """
        # Initialize various lists for the first time based on length of indices.
        if self._prev_indices is None:
            self._prev_indices = indices

            # +1 to track token counts in the last index.
            self._local_counts = np.full(len(indices) + 1, 1)
            self._local_counts[-1] = 0
            self.counts = [[] for _ in range(len(self._local_counts))]

        has_reset = False
        for i in range(len(indices)):
            # index value changed. Push all local values beyond i to count and reset those local_counts.
            # For example, if document index changed, push counts on sentences and tokens and reset their local_counts
            # to indicate that we are tracking those for new document. We need to do this at all document hierarchies.
            if indices[i] > self._prev_indices[i]:
                self._local_counts[i] += 1
                has_reset = True
                for j in range(i + 1, len(self.counts)):
                    self.counts[j].append(self._local_counts[j])
                    self._local_counts[j] = 1

        # If none of the aux indices changed, update token count.
        if not has_reset:
            self._local_counts[-1] += 1
        self._prev_indices = indices[:]