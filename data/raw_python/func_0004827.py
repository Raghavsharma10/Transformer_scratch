def count_countries(publishingCountry, **kwargs):
    '''
    Lists occurrence counts for all countries covered by the data published by the given country

    :param publishingCountry: [str] A two letter country code

    :return: dict

    Usage::

            from pygbif import occurrences
            occurrences.count_countries(publishingCountry = "DE")
    '''
    url = gbif_baseurl + 'occurrence/counts/countries'
    out = gbif_GET(url, {'publishingCountry': publishingCountry}, **kwargs)
    return out