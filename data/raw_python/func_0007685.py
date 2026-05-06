def finalize(self):
        """This will add the very last document to counts. We also get rid of counts[0] since that
        represents document level which doesnt come under anything else. We also convert all count
        values to numpy arrays so that stats can be computed easily.
        """
        for i in range(1, len(self._local_counts)):
            self.counts[i].append(self._local_counts[i])
        self.counts.pop(0)

        for i in range(len(self.counts)):
            self.counts[i] = np.array(self.counts[i])