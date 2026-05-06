def default_dist_funcs(dist_funcs, feature_example):
        """
        Fills in default distance metrics for fingerprint analyses
        """

        if dist_funcs is None:
            dist_funcs = dict()

        for key in feature_example:
            if key in dist_funcs:
                pass
            if key == 'item':
                pass
            elif isinstance(feature_example[key], (six.string_types, six.binary_type)):
                dist_funcs[key] = 'match'
            elif isinstance(feature_example[key], (int, np.integer, float)) or all([isinstance(i, (int, np.integer, float)) for i in feature_example[key]]):
                dist_funcs[key] = 'euclidean'

        return dist_funcs