def all_as_list():
    '''
    returns a list of all defined containers
    '''
    as_dict = all_as_dict()
    containers = as_dict['Running'] + as_dict['Frozen'] + as_dict['Stopped'] 
    containers_list = []
    for i in containers:
        i = i.replace(' (auto)', '')
        containers_list.append(i)
    return containers_list