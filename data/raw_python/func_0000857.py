def orcid_uri_to_orcid(value):
    "Strip the uri schema from the start of ORCID URL strings"
    if value is None:
        return value
    replace_values = ['http://orcid.org/', 'https://orcid.org/']
    for replace_value in replace_values:
        value = value.replace(replace_value, '')
    return value