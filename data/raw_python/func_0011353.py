def get_library(lookup=True, key='exp_id'):
    ''' return the raw library, without parsing'''
    library = None
    response = requests.get(EXPFACTORY_LIBRARY)
    if response.status_code == 200:
        library = response.json()
        if lookup is True:
            return make_lookup(library,key=key)
    return library