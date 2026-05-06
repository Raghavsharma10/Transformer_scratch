def build_depenses_homogenisees(temporary_store = None, year = None):
    """Build menage consumption by categorie fiscale dataframe """
    assert temporary_store is not None
    assert year is not None

    bdf_survey_collection = SurveyCollection.load(
        collection = 'budget_des_familles', config_files_directory = config_files_directory
        )
    survey = bdf_survey_collection.get_survey('budget_des_familles_{}'.format(year))

    # Homogénéisation des bases de données de dépenses

    if year == 1995:
        socioscm = survey.get_values(table = "socioscm")
        poids = socioscm[['mena', 'ponderrd', 'exdep', 'exrev']]
        # cette étape de ne garder que les données dont on est sûr de la qualité et de la véracité
        # exdep = 1 si les données sont bien remplies pour les dépenses du ménage
        # exrev = 1 si les données sont bien remplies pour les revenus du ménage
        poids = poids[(poids.exdep == 1) & (poids.exrev == 1)]
        del poids['exdep'], poids['exrev']
        poids.rename(
            columns = {
                'mena': 'ident_men',
                'ponderrd': 'pondmen',
                },
            inplace = True
            )
        poids.set_index('ident_men', inplace = True)

        conso = survey.get_values(table = "depnom")
        conso = conso[["valeur", "montant", "mena", "nomen5"]]
        conso = conso.groupby(["mena", "nomen5"]).sum()
        conso = conso.reset_index()
        conso.rename(
            columns = {
                'mena': 'ident_men',
                'nomen5': 'poste{}'.format(year),
                'valeur': 'depense',
                'montant': 'depense_avt_imput',
                },
            inplace = True
            )

        # Passage à l'euro
        conso.depense = conso.depense / 6.55957
        conso.depense_avt_imput = conso.depense_avt_imput / 6.55957
        conso_small = conso[[u'ident_men', u'poste1995', u'depense']]

        conso_unstacked = conso_small.set_index(['ident_men', 'poste1995']).unstack('poste1995')
        conso_unstacked = conso_unstacked.fillna(0)

        levels = conso_unstacked.columns.levels[1]
        labels = conso_unstacked.columns.labels[1]
        conso_unstacked.columns = levels[labels]
        conso_unstacked.rename(index = {0: 'ident_men'}, inplace = True)
        conso = conso_unstacked.merge(poids, left_index = True, right_index = True)
        conso = conso.reset_index()

    if year == 2000:
        conso = survey.get_values(table = "consomen")
        conso.rename(
            columns = {
                'ident': 'ident_men',
                'pondmen': 'pondmen',
                },
            inplace = True,
            )
        for variable in ['ctotale', 'c99', 'c99999'] + \
                        ["c0{}".format(i) for i in range(1, 10)] + \
                        ["c{}".format(i) for i in range(10, 14)]:
            del conso[variable]

    if year == 2005:
        conso = survey.get_values(table = "c05d")

    if year == 2011:
        try:
            conso = survey.get_values(table = "C05")
        except:
            conso = survey.get_values(table = "c05")
        conso.rename(
            columns = {
                'ident_me': 'ident_men',
                },
            inplace = True,
            )
        del conso['ctot']

    # Grouping by coicop

    poids = conso[['ident_men', 'pondmen']].copy()
    poids.set_index('ident_men', inplace = True)
    conso.drop('pondmen', axis = 1, inplace = True)
    conso.set_index('ident_men', inplace = True)

    matrice_passage_data_frame, selected_parametres_fiscalite_data_frame = get_transfert_data_frames(year)

    coicop_poste_bdf = matrice_passage_data_frame[['poste{}'.format(year), 'posteCOICOP']]
    coicop_poste_bdf.set_index('poste{}'.format(year), inplace = True)
    coicop_by_poste_bdf = coicop_poste_bdf.to_dict()['posteCOICOP']
    del coicop_poste_bdf

    def reformat_consumption_column_coicop(coicop):
        try:
            return int(coicop.replace('c', '').lstrip('0'))
        except:
            return numpy.NaN
    # cette étape permet d'harmoniser les df pour 1995 qui ne se présentent pas de la même façon
    # que pour les trois autres années
    if year == 1995:
        coicop_labels = [
            normalize_code_coicop(coicop_by_poste_bdf.get(poste_bdf))
            for poste_bdf in conso.columns
            ]
    else:
        coicop_labels = [
            normalize_code_coicop(coicop_by_poste_bdf.get(reformat_consumption_column_coicop(poste_bdf)))
            for poste_bdf in conso.columns
            ]
    tuples = zip(coicop_labels, conso.columns)
    conso.columns = pandas.MultiIndex.from_tuples(tuples, names=['coicop', 'poste{}'.format(year)])
    coicop_data_frame = conso.groupby(level = 0, axis = 1).sum()

    depenses = coicop_data_frame.merge(poids, left_index = True, right_index = True)

    # Création de gros postes, les 12 postes sur lesquels le calage se fera
    def select_gros_postes(coicop):
        try:
            coicop = unicode(coicop)
        except:
            coicop = coicop
        normalized_coicop = normalize_code_coicop(coicop)
        grosposte = normalized_coicop[0:2]
        return int(grosposte)

    grospostes = [
        select_gros_postes(coicop)
        for coicop in coicop_data_frame.columns
        ]
    tuples_gros_poste = zip(coicop_data_frame.columns, grospostes)
    coicop_data_frame.columns = pandas.MultiIndex.from_tuples(tuples_gros_poste, names=['coicop', 'grosposte'])

    depenses_by_grosposte = coicop_data_frame.groupby(level = 1, axis = 1).sum()
    depenses_by_grosposte = depenses_by_grosposte.merge(poids, left_index = True, right_index = True)

    # TODO : understand why it does not work: depenses.rename(columns = {u'0421': 'poste_coicop_421'}, inplace = True)

    produits = [column for column in depenses.columns if column.isdigit()]
    for code in produits:
        if code[-1:] == '0':
            depenses.rename(columns = {code: code[:-1]}, inplace = True)
        else:
            depenses.rename(columns = {code: code}, inplace = True)
    produits = [column for column in depenses.columns if column.isdigit()]
    for code in produits:
        if code[0:1] == '0':
            depenses.rename(columns = {code: code[1:]}, inplace = True)
        else:
            depenses.rename(columns = {code: code}, inplace = True)
    produits = [column for column in depenses.columns if column.isdigit()]
    for code in produits:
        depenses.rename(columns = {code: 'poste_coicop_' + code}, inplace = True)

    temporary_store['depenses_{}'.format(year)] = depenses

    depenses_by_grosposte.columns = depenses_by_grosposte.columns.astype(str)
    liste_grospostes = [column for column in depenses_by_grosposte.columns if column.isdigit()]
    for grosposte in liste_grospostes:
        depenses_by_grosposte.rename(columns = {grosposte: 'coicop12_' + grosposte}, inplace = True)

    temporary_store['depenses_by_grosposte_{}'.format(year)] = depenses_by_grosposte