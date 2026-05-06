def get_inflators(target_year):
    '''
    Fonction qui calcule les ratios de calage (bdf sur cn pour année de données) et de vieillissement
    à partir des masses de comptabilité nationale et des masses de consommation de bdf.
    '''
    data_year = find_nearest_inferior(data_years, target_year)
    inflators_bdf_to_cn = get_inflators_bdf_to_cn(data_year)
    inflators_cn_to_cn = get_inflators_cn_to_cn(target_year)

    ratio_by_variable = dict()
    for key in inflators_cn_to_cn.keys():
        ratio_by_variable[key] = inflators_bdf_to_cn[key] * inflators_cn_to_cn[key]

    return ratio_by_variable