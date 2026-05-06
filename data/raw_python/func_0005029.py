def transform_annotation(self, ann, duration):
        '''Apply the structure agreement transformation.

        Parameters
        ----------
        ann : jams.Annotation
            The segment annotation

        duration : number > 0
            The target duration

        Returns
        -------
        data : dict
            data['agree'] : np.ndarray, shape=(n, n), dtype=bool
        '''

        intervals, values = ann.to_interval_values()

        intervals, values = adjust_intervals(intervals, values,
                                             t_min=0, t_max=duration)
        # Re-index the labels
        ids, _ = index_labels(values)

        rate = float(self.hop_length) / self.sr
        # Sample segment labels on our frame grid
        _, labels = intervals_to_samples(intervals, ids, sample_size=rate)

        # Make the agreement matrix
        return {'agree': np.equal.outer(labels, labels)}