def compute_feature_weights(pres_list, rec_list, feature_list, distances):
    """
    Compute clustering scores along a set of feature dimensions

    Parameters
    ----------
    pres_list : list
        list of presented words

    rec_list : list
        list of recalled words

    feature_list : list
        list of feature dicts for presented words

    distances : dict
        dict of distance matrices for each feature

    Returns
    ----------
    weights : list
        list of clustering scores for each feature dimension
    """

    # initialize the weights object for just this list
    weights = {}
    for feature in feature_list[0]:
        weights[feature] = []

    # return default list if there is not enough data to compute the fingerprint
    if len(rec_list) <= 2:
        print('Not enough recalls to compute fingerprint, returning default'
              'fingerprint.. (everything is .5)')
        for feature in feature_list[0]:
            weights[feature] = .5
        return [weights[key] for key in weights]

    # initialize past word list
    past_words = []
    past_idxs = []

    # loop over words
    for i in range(len(rec_list)-1):

        # grab current word
        c = rec_list[i]

        # grab the next word
        n = rec_list[i + 1]

        # if both recalled words are in the encoding list and haven't been recalled before
        if (c in pres_list and n in pres_list) and (c not in past_words and n not in past_words):

            # for each feature
            for feature in feature_list[0]:

                # get the distance vector for the current word
                dists = distances[feature][pres_list.index(c),:]

                # distance between current and next word
                cdist = dists[pres_list.index(n)]

                # filter dists removing the words that have already been recalled
                dists_filt = np.array([dist for idx, dist in enumerate(dists) if idx not in past_idxs])

                # get indices
                avg_rank = np.mean(np.where(np.sort(dists_filt)[::-1] == cdist)[0]+1)

                # compute the weight
                weights[feature].append(avg_rank / len(dists_filt))

            # keep track of what has been recalled already
            past_idxs.append(pres_list.index(c))
            past_words.append(c)

    # average over the cluster scores for a particular dimension
    for feature in weights:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            weights[feature] = np.nanmean(weights[feature])

    return [weights[key] for key in weights]