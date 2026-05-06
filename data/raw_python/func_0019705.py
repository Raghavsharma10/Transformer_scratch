def order_stick(presenter, egg, dist_dict, strategy, fingerprint):
    """
    Reorders a list according to strategy
    """

    def compute_feature_stick(features, weights, alpha):
        '''create a 'stick' of feature weights'''

        feature_stick = []
        for f, w in zip(features, weights):
            feature_stick+=[f]*int(np.power(w,alpha)*100)

        return feature_stick

    def reorder_list(egg, feature_stick, dist_dict, tau):

        def compute_stimulus_stick(s, tau):
            '''create a 'stick' of feature weights'''

            feature_stick = [[weights[feature]]*round(weights[feature]**alpha)*100 for feature in w]
            return [item for sublist in feature_stick for item in sublist]

        # parse egg
        pres, rec, features, dist_funcs = parse_egg(egg)

        # turn pres and features into np arrays
        pres_arr = np.array(pres)
        features_arr = np.array(features)

        # starting with a random word
        reordered_list = []
        reordered_features = []

        # start with a random choice
        idx = np.random.choice(len(pres), 1)[0]

        # original inds
        inds = list(range(len(pres)))

        # keep track of the indices
        inds_used = [idx]

        # get the word
        current_word = pres[idx]

        # get the features dict
        current_features = features[idx]

        # append that word to the reordered list
        reordered_list.append(current_word)

        # append the features to the reordered list
        reordered_features.append(current_features)

        # loop over the word list
        for i in range(len(pres)-1):

            # sample from the stick
            feature_sample = feature_stick[np.random.choice(len(feature_stick), 1)[0]]

            # indices left
            inds_left = [ind for ind in inds if ind not in inds_used]

            # make a copy of the words filtering out the already used ones
            words_left = pres[inds_left]

            # get word distances for the word
            dists_left = np.array([dist_dict[current_word][word][feature_sample] for word in words_left])

            # features left
            features_left = features[inds_left]

            # normalize distances
            dists_left_max = np.max(dists_left)
            if dists_left_max>0:
                dists_left_norm = dists_left/np.max(dists_left)
            else:
                dists_left_norm = dists_left

            # get the min
            dists_left_min = np.min(-dists_left_norm)

            # invert the word distances to turn distance->similarity
            dists_left_inv = - dists_left_norm - dists_left_min + .01

            # create a word stick
            words_stick = []
            for word, dist in zip(words_left, dists_left_inv):
                words_stick+=[word]*int(np.power(dist,tau)*100)

            next_word = np.random.choice(words_stick)

            next_word_idx = np.where(pres==next_word)[0]

            inds_used.append(next_word_idx)

            reordered_list.append(next_word)
            reordered_features.append(features[next_word_idx][0])

        return Egg(pres=[reordered_list], rec=[reordered_list], features=[[reordered_features]], dist_funcs=dist_funcs)

    # parse egg
    pres, rec, features, dist_funcs = parse_egg(egg)

    # get params needed for list reordering
    features = presenter.get_params('fingerprint').get_features()
    alpha = presenter.get_params('alpha')
    tau = presenter.get_params('tau')
    weights = fingerprint

    # invert the weights if strategy is destabilize
    if strategy=='destabilize':
        weights = 1 - weights

    # compute feature stick
    feature_stick = compute_feature_stick(features, weights, alpha)

    # reorder list
    return reorder_list(egg, feature_stick, dist_dict, tau)