def get_inflators_cn_to_cn(target_year):
    '''
        Calcule l'inflateur de vieillissement à partir des masses de comptabilité nationale.
    '''
    data_year = find_nearest_inferior(data_years, target_year)
    data_year_cn_aggregates = get_cn_aggregates(data_year)['consoCN_COICOP_{}'.format(data_year)].to_dict()
    target_year_cn_aggregates = get_cn_aggregates(target_year)['consoCN_COICOP_{}'.format(target_year)].to_dict()

    return dict(
        (key, target_year_cn_aggregates[key] / data_year_cn_aggregates[key])
        for key in data_year_cn_aggregates.keys()
        )