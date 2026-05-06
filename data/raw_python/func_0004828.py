def count_publishingcountries(country, **kwargs):
    '''
    Lists occurrence counts for all countries that publish data about the given country

    :param country: [str] A country, two letter code

    :return: dict

    Usage::

            from pygbif import occurrences
            occurrences.count_publishingcountries(country = "DE")
    '''
    url = gbif_baseurl + 'occurrence/counts/publishingCountries'
    out = gbif_GET(url, {"country": country}, **kwargs)
    return out