def load_builtin_slots():
    '''
    Helper function to load builtin slots from the data location
    '''
    builtin_slots = {}
    for index, line in enumerate(open(BUILTIN_SLOTS_LOCATION)):
        o =  line.strip().split('\t')
        builtin_slots[index] = {'name' : o[0],
                                'description' : o[1] } 
    return builtin_slots