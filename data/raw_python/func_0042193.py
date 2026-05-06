def get_bdf_data_frames(depenses, year_data = None):
    assert year_data is not None
    '''
    Récupère les dépenses de budget des familles et les agrège par poste
    (en tenant compte des poids respectifs des ménages)
    '''
    depenses_by_grosposte = pandas.DataFrame()
    for grosposte in range(1, 13):
        if depenses_by_grosposte is None:
            depenses_by_grosposte = depenses['coicop12_{}'.format(grosposte)]
        else:
            depenses_by_grosposte = concat([depenses_by_grosposte, depenses['coicop12_{}'.format(grosposte)]], axis = 1)
    depenses_by_grosposte = concat([depenses_by_grosposte, depenses['pondmen']], axis = 1)
    grospostes_list = set(depenses_by_grosposte.columns)
    grospostes_list.remove('pondmen')

    dict_bdf_weighted_sum_by_grosposte = {}
    for grosposte in grospostes_list:
        depenses_by_grosposte['{}pond'.format(grosposte)] = (
            depenses_by_grosposte[grosposte] * depenses_by_grosposte['pondmen']
            )
        dict_bdf_weighted_sum_by_grosposte[grosposte] = depenses_by_grosposte['{}pond'.format(grosposte)].sum()
    df_bdf_weighted_sum_by_grosposte = pandas.DataFrame(
        pandas.Series(
            data = dict_bdf_weighted_sum_by_grosposte,
            index = dict_bdf_weighted_sum_by_grosposte.keys()
            )
        )
    return df_bdf_weighted_sum_by_grosposte