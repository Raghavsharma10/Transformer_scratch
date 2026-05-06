def calcul_ratios_calage(year_data, year_calage, data_bdf, data_cn):
    '''
    Fonction qui calcule les ratios de calage (bdf sur cn pour année de données) et de vieillissement
    à partir des masses de comptabilité nationale et des masses de consommation de bdf.
    '''
    masses = data_cn.merge(
        data_bdf, left_index = True, right_index = True
        )
    masses.rename(columns = {0: 'conso_bdf{}'.format(year_data)}, inplace = True)
    if year_calage != year_data:
        masses['ratio_cn{}_cn{}'.format(year_data, year_calage)] = (
            masses['consoCN_COICOP_{}'.format(year_calage)] / masses['consoCN_COICOP_{}'.format(year_data)]
            )
    if year_calage == year_data:
        masses['ratio_cn{}_cn{}'.format(year_data, year_calage)] = 1

    masses['ratio_bdf{}_cn{}'.format(year_data, year_data)] = (
        1e6 * masses['consoCN_COICOP_{}'.format(year_data)] / masses['conso_bdf{}'.format(year_data)]
        )
    return masses