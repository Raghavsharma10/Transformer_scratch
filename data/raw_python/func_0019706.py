def stick_perm(presenter, egg, dist_dict, strategy):
    """Computes weights for one reordering using stick-breaking method"""

    # seed RNG
    np.random.seed()

    # unpack egg
    egg_pres, egg_rec, egg_features, egg_dist_funcs = parse_egg(egg)

    # reorder
    regg = order_stick(presenter, egg, dist_dict, strategy)

    # unpack regg
    regg_pres, regg_rec, regg_features, regg_dist_funcs = parse_egg(regg)

    # # get the order
    regg_pres = list(regg_pres)
    egg_pres = list(egg_pres)
    idx = [egg_pres.index(r) for r in regg_pres]

    # compute weights
    weights = compute_feature_weights_dict(list(regg_pres), list(regg_pres), list(regg_features), dist_dict)

    # save out the order
    orders = idx

    return weights, orders