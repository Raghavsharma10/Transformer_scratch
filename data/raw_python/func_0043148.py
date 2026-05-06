def get_inflators_bdf_to_cn(data_year):
    '''
    Calcule les ratios de calage (bdf sur cn pour année de données)
    à partir des masses de comptabilité nationale et des masses de consommation de bdf.
    '''
    data_cn = get_cn_aggregates(data_year)
    data_bdf = get_bdf_aggregates(data_year)
    masses = data_cn.merge(
        data_bdf, left_index = True, right_index = True
        )
    masses.rename(columns = {'bdf_aggregates': 'conso_bdf{}'.format(data_year)}, inplace = True)
    return (
        masses['consoCN_COICOP_{}'.format(data_year)] / masses['conso_bdf{}'.format(data_year)]
        ).to_dict()