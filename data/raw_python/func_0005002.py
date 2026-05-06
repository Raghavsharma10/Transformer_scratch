def merge(self, data):
        '''Merge an array of output dictionaries into a single dictionary
        with properly scoped names.

        Parameters
        ----------
        data : list of dict
            Output dicts as produced by `pumpp.task.BaseTaskTransformer.transform`
            or `pumpp.feature.FeatureExtractor.transform`.

        Returns
        -------
        data_out : dict
            All elements of the input dicts are stacked along the 0 axis,
            and keys are re-mapped by `scope`.
        '''
        data_out = dict()

        # Iterate over all keys in data
        for key in set().union(*data):
            data_out[self.scope(key)] = np.stack([np.asarray(d[key]) for d in data],
                                                 axis=0)
        return data_out