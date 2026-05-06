def count_year(year, **kwargs):
    '''
    Lists occurrence counts by year

    :param year: [int] year range, e.g., ``1990,2000``. Does not support ranges like ``asterisk,2010``

    :return: dict

    Usage::

            from pygbif import occurrences
            occurrences.count_year(year = '1990,2000')
    '''
    url = gbif_baseurl + 'occurrence/counts/year'
    out = gbif_GET(url, {'year': year}, **kwargs)
    return out