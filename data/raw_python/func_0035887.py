def delta13c_constants():
    """
    Constants for calculating delta13C values from ratios.
    From website of Verkouteren & Lee 2001 Anal. Chem.
    """
    # possible values for constants (from NIST)
    cst = OrderedDict()
    cst['Craig'] = {'S13': 0.0112372, 'S18': 0.002079,
                    'K': 0.008333, 'A': 0.5}
    cst['IAEA'] = {'S13': 0.0112372, 'S18': 0.00206716068,
                   'K': 0.0091993, 'A': 0.5}
    cst['Werner'] = {'S13': 0.0112372, 'S18': 0.0020052,
                     'K': 0.0093704, 'A': 0.516}
    cst['Santrock'] = {'S13': 0.0112372, 'S18': 0.0020052,
                       'K': 0.0099235, 'A': 0.516}
    cst['Assonov'] = {'S13': 0.0112372, 'S18': 0.0020052,
                      'K': 0.0102819162, 'A': 0.528}
    cst['Assonov2'] = {'S13': 0.0111802, 'S18': 0.0020052,
                       'K': 0.0102819162, 'A': 0.528}
    cst['Isodat'] = {'S13': 0.0111802, 'S18': 0.0020052,
                     'K': 0.0099235, 'A': 0.516}
    return cst