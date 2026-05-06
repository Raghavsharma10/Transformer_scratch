def ceres(data_set='ceres'):
    """Twenty two observations of the Dwarf planet Ceres as observed by Giueseppe Piazzi and published in the September edition of Monatlicher Correspondenz in 1801. These were the measurements used by Gauss to fit a model of the planets orbit through which the planet was recovered three months later."""
    if not data_available(data_set):
        download_data(data_set)
    import pandas as pd
    data = pd.read_csv(os.path.join(data_path, data_set, 'ceresData.txt'), index_col = 'Tag', header=None, sep='\t',names=['Tag', 'Mittlere Sonnenzeit', 'Gerade Aufstig in Zeit', 'Gerade Aufstiegung in Graden', 'Nordlich Abweich', 'Geocentrische Laenger', 'Geocentrische Breite', 'Ort der Sonne + 20" Aberration', 'Logar. d. Distanz'], parse_dates=True, dayfirst=False)
    return data_details_return({'data': data}, data_set)