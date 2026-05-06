def count(taxonKey=None, basisOfRecord=None, country=None, isGeoreferenced=None,
    datasetKey=None, publishingCountry=None, typeStatus=None,
    issue=None, year=None, **kwargs):
    '''
    Returns occurrence counts for a predefined set of dimensions

    :param taxonKey: [int] A GBIF occurrence identifier
    :param basisOfRecord: [str] A GBIF occurrence identifier
    :param country: [str] A GBIF occurrence identifier
    :param isGeoreferenced: [bool] A GBIF occurrence identifier
    :param datasetKey: [str] A GBIF occurrence identifier
    :param publishingCountry: [str] A GBIF occurrence identifier
    :param typeStatus: [str] A GBIF occurrence identifier
    :param issue: [str] A GBIF occurrence identifier
    :param year: [int] A GBIF occurrence identifier

    :return: dict

    Usage::

        from pygbif import occurrences
        occurrences.count(taxonKey = 3329049)
        occurrences.count(country = 'CA')
        occurrences.count(isGeoreferenced = True)
        occurrences.count(basisOfRecord = 'OBSERVATION')
    '''
    url = gbif_baseurl + 'occurrence/count'
    out = gbif_GET(url, {'taxonKey': taxonKey, 'basisOfRecord': basisOfRecord, 'country': country,
        'isGeoreferenced': isGeoreferenced, 'datasetKey': datasetKey,
        'publishingCountry': publishingCountry, 'typeStatus': typeStatus,
        'issue': issue, 'year': year}, **kwargs)
    return out