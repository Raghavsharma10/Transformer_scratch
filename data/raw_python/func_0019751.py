def parse_egg(egg):
    """Parses an egg and returns fields"""
    pres_list = egg.get_pres_items().values[0]
    rec_list = egg.get_rec_items().values[0]
    feature_list = egg.get_pres_features().values[0]
    dist_funcs = egg.dist_funcs
    return pres_list, rec_list, feature_list, dist_funcs