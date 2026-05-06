def doi_uri_to_doi(value):
    "Strip the uri schema from the start of DOI URL strings"
    if value is None:
        return value
    replace_values = ['http://dx.doi.org/', 'https://dx.doi.org/',
                      'http://doi.org/', 'https://doi.org/']
    for replace_value in replace_values:
        value = value.replace(replace_value, '')
    return value