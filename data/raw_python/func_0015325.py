def _split_license(license):
    '''Returns all individual licenses in the input'''
    return (x.strip() for x in (l for l in _regex.split(license) if l))