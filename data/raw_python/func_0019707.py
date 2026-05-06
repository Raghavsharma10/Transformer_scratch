def compute_distances_dict(egg):
    """ Creates a nested dict of distances """
    pres, rec, features, dist_funcs = parse_egg(egg)
    pres_list = list(pres)
    features_list = list(features)

    # initialize dist dict
    distances = {}

    # for each word in the list
    for idx1, item1 in enumerate(pres_list):

        distances[item1]={}

        # for each word in the list
        for idx2, item2 in enumerate(pres_list):

            distances[item1][item2]={}

            # for each feature in dist_funcs
            for feature in dist_funcs:
                distances[item1][item2][feature] = builtin_dist_funcs[dist_funcs[feature]](features_list[idx1][feature],features_list[idx2][feature])

    return distances