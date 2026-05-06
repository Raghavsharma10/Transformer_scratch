def count_datasets(taxonKey = None, country = None, **kwargs):
    '''
    Lists occurrence counts for datasets that cover a given taxon or country

    :param taxonKey: [int] Taxon key
    :param country: [str] A country, two letter code

    :return: dict

    Usage::

            from pygbif import occurrences
            occurrences.count_datasets(country = "DE")
    '''
    url = gbif_baseurl + 'occurrence/counts/datasets'
    out = gbif_GET(url, {'taxonKey': taxonKey, 'country': country}, **kwargs)
    return out