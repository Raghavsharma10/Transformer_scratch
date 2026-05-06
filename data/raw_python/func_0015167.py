def _unpaginated(what):
    '''Returns a dictionary with all <what>, unpaginated'''
    page = data(what)
    results = page['results']
    count = page['count']
    while page['next']:
        page = data(page['next'])
        results += page['results']
        count += page['count']
    return {'results': results, 'count': count}