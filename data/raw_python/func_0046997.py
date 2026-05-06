def read_features_and_groups(features_path, groups_path):
    "Reader for data and groups"

    try:
        if not pexists(features_path):
            raise ValueError('non-existent features file')

        if not pexists(groups_path):
            raise ValueError('non-existent groups file')

        if isinstance(features_path, str):
            features = np.genfromtxt(features_path, dtype=float)
        else:
            raise ValueError('features input must be a file path ')

        if isinstance(groups_path, str):
            groups = np.genfromtxt(groups_path, dtype=str)
        else:
            raise ValueError('groups input must be a file path ')

    except:
        raise IOError('error reading the specified features and/or groups.')

    if len(features) != len(groups):
        raise ValueError("lengths of features and groups do not match!")

    return features, groups