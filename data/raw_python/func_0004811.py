def download(queries, user=None, pwd=None,
             email=None, pred_type='and'):
    """
    Spin up a download request for GBIF occurrence data.

    :param queries: One or more of query arguments to kick of a download job.
        See Details.
    :type queries: str or list
    :param pred_type: (character) One of ``equals`` (``=``), ``and`` (``&``),
        `or`` (``|``), ``lessThan`` (``<``), ``lessThanOrEquals`` (``<=``),
        ``greaterThan`` (``>``), ``greaterThanOrEquals`` (``>=``),
        ``in``, ``within``, ``not`` (``!``), ``like``
    :param user: (character) User name within GBIF's website.
        Required. Set in your env vars with the option ``GBIF_USER``
    :param pwd: (character) User password within GBIF's website. Required.
        Set in your env vars with the option ``GBIF_PWD``
    :param email: (character) Email address to recieve download notice done
        email. Required. Set in your env vars with the option ``GBIF_EMAIL``

    Argument passed have to be passed as character (e.g., ``country = US``),
    with a space between key (``country``), operator (``=``), and value (``US``).
    See the ``type`` parameter for possible options for the operator.
    This character string is parsed internally.

    Acceptable arguments to ``...`` (args) are:

     - taxonKey = ``TAXON_KEY``
     - scientificName = ``SCIENTIFIC_NAME``
     - country = ``COUNTRY``
     - publishingCountry = ``PUBLISHING_COUNTRY``
     - hasCoordinate = ``HAS_COORDINATE``
     - hasGeospatialIssue = ``HAS_GEOSPATIAL_ISSUE``
     - typeStatus = ``TYPE_STATUS``
     - recordNumber = ``RECORD_NUMBER``
     - lastInterpreted = ``LAST_INTERPRETED``
     - continent = ``CONTINENT``
     - geometry = ``GEOMETRY``
     - basisOfRecord = ``BASIS_OF_RECORD``
     - datasetKey = ``DATASET_KEY``
     - eventDate = ``EVENT_DATE``
     - catalogNumber = ``CATALOG_NUMBER``
     - year = ``YEAR``
     - month = ``MONTH``
     - decimalLatitude = ``DECIMAL_LATITUDE``
     - decimalLongitude = ``DECIMAL_LONGITUDE``
     - elevation = ``ELEVATION``
     - depth = ``DEPTH``
     - institutionCode = ``INSTITUTION_CODE``
     - collectionCode = ``COLLECTION_CODE``
     - issue = ``ISSUE``
     - mediatype = ``MEDIA_TYPE``
     - recordedBy = ``RECORDED_BY``
     - repatriated = ``REPATRIATED``

    See the API docs http://www.gbif.org/developer/occurrence#download
    for more info, and the predicates docs
    http://www.gbif.org/developer/occurrence#predicates

    GBIF has a limit of 12,000 characters for download queries - so
    if you're download request is really, really long and complex,
    consider breaking it up into multiple requests by one factor or
    another.

    :return: A dictionary, of results

    Usage::

        from pygbif import occurrences as occ

        occ.download('basisOfRecord = LITERATURE')
        occ.download('taxonKey = 3119195')
        occ.download('decimalLatitude > 50')
        occ.download('elevation >= 9000')
        occ.download('decimalLatitude >= 65')
        occ.download('country = US')
        occ.download('institutionCode = TLMF')
        occ.download('catalogNumber = Bird.27847588')

        res = occ.download(['taxonKey = 7264332', 'hasCoordinate = TRUE'])

        # pass output to download_meta for more information
        occ.download_meta(occ.download('decimalLatitude > 75'))

        # Multiple queries
        gg = occ.download(['decimalLatitude >= 65',
                          'decimalLatitude <= -65'], type='or')
        gg = occ.download(['depth = 80', 'taxonKey = 2343454'],
                          type='or')

        # Repratriated data for Costa Rica
        occ.download(['country = CR', 'repatriated = true'])
    """

    user = _check_environ('GBIF_USER', user)
    pwd = _check_environ('GBIF_PWD', pwd)
    email = _check_environ('GBIF_EMAIL', email)

    if isinstance(queries, str):
        queries = [queries]

    keyval = [_parse_args(z) for z in queries]

    # USE GBIFDownload class to set up the predicates
    req = GbifDownload(user, email)
    req.main_pred_type = pred_type
    for predicate in keyval:
        req.add_predicate(predicate['key'],
                          predicate['value'],
                          predicate['type'])

    out = req.post_download(user, pwd)
    return out, req.payload