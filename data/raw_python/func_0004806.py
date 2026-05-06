def occ_issues_lookup(issue=None, code=None):
    '''
    Lookup occurrence issue definitions and short codes

    :param issue: Full name of issue, e.g, CONTINENT_COUNTRY_MISMATCH
    :param code: an issue short code, e.g. ccm

    Usage
    pygbif.occ_issues_lookup(issue = 'CONTINENT_COUNTRY_MISMATCH')
    pygbif.occ_issues_lookup(issue = 'MULTIMEDIA_DATE_INVALID')
    pygbif.occ_issues_lookup(issue = 'ZERO_COORDINATE')
    pygbif.occ_issues_lookup(code = 'cdiv')
    '''
    if code is None:
        bb = [trymatch(issue, x) for x in gbifissues['issue'] ]
        tmp = filter(None, bb)
    else:
        bb = [trymatch(code, x) for x in gbifissues['code'] ]
        tmp = filter(None, bb)
    return tmp