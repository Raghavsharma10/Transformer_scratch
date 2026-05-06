def normalize_code_coicop(code):
    '''Normalize_coicop est function d'harmonisation de la colonne d'entiers posteCOICOP de la table
matrice_passage_data_frame en la transformant en une chaine de 5 caractères afin de pouvoir par la suite agréger les postes
COICOP selon les 12 postes agrégés de la nomenclature de la comptabilité nationale. Chaque poste contient 5 caractères,
les deux premiers (entre 01 et 12) correspondent à ces postes agrégés de la CN.

    '''
    # TODO: vérifier la formule !!!

    try:
        code = unicode(code)
    except:
        code = code
    if len(code) == 3:
        code_coicop = "0" + code + "0"  # "{0}{1}{0}".format(0, code)
    elif len(code) == 4:
        if not code.startswith("0") and not code.startswith("1") and not code.startswith("45") and not code.startswith("9"):
            code_coicop = "0" + code
            # 022.. = cigarettes et tabacs => on les range avec l'alcool (021.0)
        elif code.startswith("0"):
            code_coicop = code + "0"
        elif code in ["1151", "1181", "4552", "4522", "4511", "9122", "9151", "9211", "9341", "1411"]:
            # 1151 = Margarines et autres graisses végétales
            # 1181 = Confiserie
            # 04522 = Achat de butane, propane
            # 04511 = Facture EDF GDF non dissociables
            code_coicop = "0" + code
        else:
            # 99 = loyer, impots et taxes, cadeaux...
            code_coicop = code + "0"
    elif len(code) == 5:
        if not code.startswith("13") and not code.startswith("44") and not code.startswith("51"):
            code_coicop = code
        else:
            code_coicop = "99000"
    else:
        log.error("Problematic code {}".format(code))
        raise()
    return code_coicop