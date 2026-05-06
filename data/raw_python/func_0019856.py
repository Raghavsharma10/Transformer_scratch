def recall_matrix(egg, match='exact', distance='euclidean', features=None):
    """
    Computes recall matrix given list of presented and list of recalled words

    Parameters
    ----------
    egg : quail.Egg
        Data to analyze

    match : str (exact, best or smooth)
        Matching approach to compute recall matrix.  If exact, the presented and
        recalled items must be identical (default).  If best, the recalled item
        that is most similar to the presented items will be selected. If smooth,
        a weighted average of all presented items will be used, where the
        weights are derived from the similarity between the recalled item and
        each presented item.

    distance : str
        The distance function used to compare presented and recalled items.
        Applies only to 'best' and 'smooth' matching approaches.  Can be any
        distance function supported by numpy.spatial.distance.cdist.

    Returns
    ----------
    recall_matrix : list of lists of ints
      each integer represents the presentation position of the recalled word in a given list in order of recall
      0s represent recalled words not presented
      negative ints represent words recalled from previous lists

    """

    if match in ['best', 'smooth']:
        if not features:
            features = [k for k,v in egg.pres.loc[0][0].values[0].items() if k!='item']
            if not features:
                raise('No features found.  Cannot match with best or smooth strategy')

    if not isinstance(features, list):
        features = [features]

    if match=='exact':
        features=['item']
        return _recmat_exact(egg.pres, egg.rec, features)
    else:
        return _recmat_smooth(egg.pres, egg.rec, features, distance, match)