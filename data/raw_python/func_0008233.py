def checkSimbad(g, target, maxobj=5, timeout=5):
    """
    Sends off a request to Simbad to check whether a target is recognised.
    Returns with a list of results, or raises an exception if it times out
    """
    url = 'http://simbad.u-strasbg.fr/simbad/sim-script'
    q = 'set limit ' + str(maxobj) + \
        '\nformat object form1 "Target: %IDLIST(1) | %COO(A D;ICRS)"\nquery ' \
        + target
    query = urllib.parse.urlencode({'submit': 'submit script', 'script': q})
    resp = urllib.request.urlopen(url, query.encode(), timeout)
    data = False
    error = False
    results = []
    for line in resp:
        line = line.decode()
        if line.startswith('::data::'):
            data = True
        if line.startswith('::error::'):
            error = True
        if data and line.startswith('Target:'):
            name, coords = line[7:].split(' | ')
            results.append(
                {'Name': name.strip(), 'Position': coords.strip(),
                 'Frame': 'ICRS'})
    resp.close()

    if error and len(results):
        g.clog.warn('drivers.check: Simbad: there appear to be some ' +
                    'results but an error was unexpectedly raised.')
    return results